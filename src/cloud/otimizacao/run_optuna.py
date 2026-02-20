
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import polars as pl
import numpy as np
import yaml
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to sys.path to allow importing from src/
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.models.model import QuantGodModel

# Logger setup — FileHandler is added dynamically in run_optimization()
# after parsing the config, so log filename includes labelled_dir suffix + timestamp.
log_dir = Path("logs/optimization")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # File handler added dynamically
)
logger = logging.getLogger(__name__)

def _setup_log_file(labelled_dir: str):
    """
    Adds a FileHandler to the root logger using the labelled_dir suffix + timestamp.
    Example: optimization_SELL_0004_BUY_0004_1h_20260220_014500.log
    labelled_dir path example: 'data/L2/labelled/labelled_SELL_0004_BUY_0004_1h'
    """
    # Extract suffix: 'labelled_SELL_0004_BUY_0004_1h' → 'SELL_0004_BUY_0004_1h'
    dir_name = Path(labelled_dir).name
    suffix = dir_name.replace("labelled_", "") if dir_name.startswith("labelled_") else dir_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"optimization_{suffix}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_filename, mode='w')  # 'w' = new file each run
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {log_filename}")

class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset eficiente que gera sequências sob demanda, economizando memória RAM.
    """
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_label = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.long)

def load_data(labelled_dir):
    parquet_files = sorted(list(Path(labelled_dir).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No labelled data found in {labelled_dir}")
    
    feature_cols = [
        'body', 'upper_wick', 'lower_wick', 'log_ret_close',
        'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
    ]
    
    dfs = []
    for pf in parquet_files:
        dfs.append(pl.read_parquet(pf, columns=feature_cols + ['target']))
    
    df = pl.concat(dfs)
    return df, feature_cols

def objective(trial, X_all, y_all, config):
    # #10: Wrap in try-except for OOM resilience
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Search Space ---
        d_model = trial.suggest_categorical("d_model", config['search_space']['d_model'])
        
        # #8: nhead must be a divisor of d_model (mathematically valid combinations only)
        valid_nheads = [h for h in config['search_space']['nhead'] if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", valid_nheads)
        
        # #7: num_layers as suggest_int (smarter TPE search vs categorical)
        num_layers = trial.suggest_int("num_layers", 
                                       min(config['search_space']['num_layers']), 
                                       max(config['search_space']['num_layers']), 
                                       step=2)
        
        batch_size = trial.suggest_categorical("batch_size", config['search_space']['batch_size'])
        dropout = trial.suggest_categorical("dropout", config['search_space']['dropout'])
        seq_len = trial.suggest_categorical("seq_len", config['search_space']['seq_len'])
        lr = trial.suggest_float("lr", config['search_space']['lr'][0], config['search_space']['lr'][1], log=True)
        
        epochs = config['search_space']['epochs']
        
        logger.info(f"Trial {trial.number} START | d_model={d_model}, nhead={nhead}, layers={num_layers}, "
                    f"batch={batch_size}, seq_len={seq_len}, drop={dropout}, lr={lr:.6f}")
        
        # --- Data Split (Chronological 80/20) ---
        split_idx = int(len(X_all) * 0.8)
        X_train_raw, y_train_raw = X_all[:split_idx], y_all[:split_idx]
        X_val_raw, y_val_raw = X_all[split_idx:], y_all[split_idx:]
        
        train_dataset = SequenceDataset(X_train_raw, y_train_raw, seq_len)
        val_dataset = SequenceDataset(X_val_raw, y_val_raw, seq_len)
        
        # num_workers=4: safe for multiprocessing (persistent_workers removed to prevent
        # stale worker PID crashes when DataLoader is deleted during OOM handling)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        # --- Model ---
        model = QuantGodModel(
            num_features=X_all.shape[1],
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=3,
            dropout=dropout
        ).to(DEVICE)
        
        # Weights for CrossEntropyLoss to handle class imbalance (SELL=0, NEUTRAL=1, BUY=2)
        # Sells and Buys are 2x more important than Neutrals
        weights = torch.tensor([2.0, 1.0, 2.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        # #3: AdamW with weight_decay for proper Transformer regularization
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # #4: LR Scheduler for better convergence in 5 epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        amp_scaler = torch.amp.GradScaler('cuda')
        
        # --- Training Loop ---
        best_val_f1 = 0
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            batch_idx = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                
                train_loss += loss.item()
                batch_idx += 1
                if batch_idx % 100 == 0:
                    percent = (batch_idx / len(train_loader)) * 100
                    logger.info(f"Trial {trial.number} | Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} ({percent:.1f}%) | Loss: {loss.item():.4f}")
            
            # Step scheduler after each epoch
            scheduler.step()
            
            # --- Validation with autocast (#5) ---
            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                with torch.amp.autocast('cuda'):  # #5: AMP on validation too
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                        outputs = model(batch_X)
                        loss_v = criterion(outputs, batch_y)
                        val_loss += loss_v.item()
                        preds = torch.argmax(outputs, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(batch_y.cpu().numpy())
            
            f1_weighted = f1_score(all_targets, all_preds, average='weighted')
            f1_macro = f1_score(all_targets, all_preds, average='macro')
            acc = accuracy_score(all_targets, all_preds)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            current_lr = scheduler.get_last_lr()[0]
            
            logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                        f"F1 Weighted: {f1_weighted:.4f} | F1 Macro: {f1_macro:.4f} | "
                        f"Acc: {acc:.4f} | LR: {current_lr:.6f}")

            if f1_weighted > best_val_f1:
                best_val_f1 = f1_weighted
                
            trial.report(f1_weighted, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                # #9: Clean up before pruning
                del model, train_loader, val_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
        
        # #9: Explicit VRAM cleanup after each trial
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        
        return best_val_f1

    except RuntimeError as e:
        # #10: Catch OOM and prune trial instead of crashing the whole study
        if "out of memory" in str(e).lower():
            logger.warning(f"Trial {trial.number} — OOM! Pruning and continuing...")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        raise e

def run_optimization():
    # 1. Config
    config_path = Path("src/cloud/otimizacao/optimization_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup log file with dynamic name (labelled_dir suffix + timestamp)
    _setup_log_file(config['paths']['labelled_dir'])
    
    # 2. Data
    logger.info("Loading data for optimization...")
    df, feature_cols = load_data(config['paths']['labelled_dir'])
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_all = df.select('target').to_numpy().flatten().astype(np.int64)
    logger.info(f"Data loaded: {X_raw.shape}")
    
    # 3. Normalization (Fit on Train Only) — float32 to halve RAM usage (#6 from report)
    split_idx = int(len(X_raw) * 0.8)
    logger.info("Fitting Scaler on Training Split (First 80%) for Optimization...")
    scaler = StandardScaler()
    scaler.fit(X_raw[:split_idx])
    X_all = scaler.transform(X_raw).astype(np.float32)

    # 4. Study
    study = optuna.create_study(
        study_name=config['paths']['study_name'],
        storage=config['paths']['db_path'],
        direction="maximize",
        load_if_exists=True
    )
    
    logger.info(f"Starting {config['optimization']['n_trials']} trials...")
    study.optimize(
        lambda trial: objective(trial, X_all, y_all, config),
        n_trials=config['optimization']['n_trials'],
        timeout=config['optimization']['timeout']
    )
    
    logger.info("Optimization finished.")
    logger.info(f"Best trial value: {study.best_trial.value}")
    logger.info(f"Best parameters: {study.best_params}")
    
    with open("src/cloud/otimizacao/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info("Best parameters saved to cloud/otimizacao/best_params.json")

if __name__ == "__main__":
    run_optimization()
