
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import os

# Add project root to sys.path to allow importing from src/
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.models.model import QuantGodModel

# Logger setup
log_dir = Path("logs/optimization")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "optimization_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
        # Retorna a sequência [idx : idx+seq_len] e o label na posição idx+seq_len-1
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
    # Search Space - Using suggest_categorical for specific lists requested by user
    d_model = trial.suggest_categorical("d_model", config['search_space']['d_model'])
    nhead = trial.suggest_categorical("nhead", config['search_space']['nhead'])
    num_layers = trial.suggest_categorical("num_layers", config['search_space']['num_layers'])
    batch_size = trial.suggest_categorical("batch_size", config['search_space']['batch_size'])
    dropout = trial.suggest_categorical("dropout", config['search_space']['dropout'])
    seq_len = trial.suggest_categorical("seq_len", config['search_space']['seq_len'])
    
    # LR is still suggested as a float range
    lr = trial.suggest_float("lr", config['search_space']['lr'][0], config['search_space']['lr'][1], log=True)
    
    epochs = config['search_space']['epochs']
    
    # Simple Train/Val Split (80/20) - Chronological
    split_idx = int(len(X_all) * 0.8)
    X_train_raw, y_train_raw = X_all[:split_idx], y_all[:split_idx]
    X_val_raw, y_val_raw = X_all[split_idx:], y_all[split_idx:]
    
    # Datasets sob demanda (Não consomem RAM extra)
    train_dataset = SequenceDataset(X_train_raw, y_train_raw, seq_len)
    val_dataset = SequenceDataset(X_val_raw, y_val_raw, seq_len)
    
    # Tensors
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = QuantGodModel(
        num_features=X_all.shape[1],
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=3,
        dropout=dropout
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loop
    best_val_f1 = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                
                loss_v = criterion(outputs, batch_y)
                val_loss += loss_v.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate Metrics
        from sklearn.metrics import f1_score, accuracy_score
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        acc = accuracy_score(all_targets, all_preds)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                    f"F1 Weighted: {f1_weighted:.4f} | F1 Macro: {f1_macro:.4f} | Acc: {acc:.4f}")

        if f1_weighted > best_val_f1:
            best_val_f1 = f1_weighted
            
        # Trial report for pruning
        trial.report(f1_weighted, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()
            
    return best_val_f1

def run_optimization():
    # 1. Config
    config_path = Path("src/cloud/otimizacao/optimization_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Data
    logger.info("Loading data for optimization...")
    df, feature_cols = load_data(config['paths']['labelled_dir'])
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_all = df.select('target').to_numpy().flatten().astype(np.int64)
    logger.info(f"Data loaded: {X_raw.shape}")
    
    # 3. Normalization (Fit on Train Only)
    split_idx = int(len(X_raw) * 0.8)
    from sklearn.preprocessing import StandardScaler
    logger.info("Fitting Scaler on Training Split (First 80%) for Optimization...")
    scaler = StandardScaler()
    scaler.fit(X_raw[:split_idx])
    X_all = scaler.transform(X_raw)

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
    
    # Save best params to a file for training
    import json
    with open("src/cloud/otimizacao/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info("Best parameters saved to cloud/otimizacao/best_params.json")

if __name__ == "__main__":
    run_optimization()
