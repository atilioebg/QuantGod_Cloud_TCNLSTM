import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import polars as pl
import numpy as np
import yaml
import logging
import pickle
from datetime import datetime
from pathlib import Path
import json
import sys
import optuna

# Project root on sys.path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing
from sklearn.preprocessing import StandardScaler
from src.cloud.base_model.utils.logging_utils import setup_logger, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import get_specialized_dir
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """Memory-efficient demand-based sequence generator with Island Split protection."""
    def __init__(self, X: np.ndarray, y: np.ndarray, island_ids: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.island_ids = island_ids
        
        # Pre-calculate valid indices that don't cross islands
        # An index is valid if island_ids[idx] == island_ids[idx + seq_len - 1]
        # We also check the total length
        max_idx = len(X) - seq_len
        if max_idx < 0:
            self.valid_indices = []
        else:
            # Vectorized check for speed
            self.valid_indices = np.where(island_ids[:max_idx + 1] == island_ids[seq_len - 1:])[0]
        
        logger.info(f"SequenceDataset: {len(self.valid_indices)}/{max_idx + 1 if max_idx >= 0 else 0} valid sequences (Lookback protection active).")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x_seq = self.X[real_idx: real_idx + self.seq_len]
        y_label = self.y[real_idx + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.long)

def load_config():
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_symmetric_space(original_space, chosen_val, is_float=False):
    """
    Applies the Symmetric Delta rule. If the chosen value from foundation is an extremum, 
    it generates a symmetric point outside the original bounds to guarantee 3 options.
    If it's a float, generates -15%, exact, +15%.
    """
    if is_float:
        return [chosen_val * 0.85, chosen_val, chosen_val * 1.15]
    
    if chosen_val not in original_space:
        original_space = sorted(original_space + [chosen_val])
        
    idx = original_space.index(chosen_val)
    if idx > 0 and idx < len(original_space) - 1:
        # Center element
        return [original_space[idx-1], chosen_val, original_space[idx+1]]
    elif idx == 0:
        # Lower extremum
        if len(original_space) > 1:
            delta = original_space[1] - chosen_val
        else:
            delta = int(chosen_val * 0.5) if isinstance(chosen_val, int) else chosen_val * 0.5
        new_val = chosen_val - delta
        # Ensure sizes like batch_size or seq_len don't go below 1 or negatives
        if isinstance(chosen_val, int):
            new_val = max(1, int(new_val))
        return [new_val, chosen_val, chosen_val + delta]
    else:
        # Upper extremum
        delta = chosen_val - original_space[idx-1]
        new_val = chosen_val + delta
        if isinstance(chosen_val, int):
            new_val = int(new_val)
        return [chosen_val - delta, chosen_val, new_val]

def build_specialist_space(config, best_foundation_params):
    """Constructs the restricted search space based on foundation params."""
    orig_space = config['optimization']['search_space']
    spec_space = {}
    
    # Categorical/Integers
    int_keys = ['tcn_channels', 'lstm_hidden', 'num_lstm_layers', 'batch_size', 'seq_len']
    for k in int_keys:
        chosen = best_foundation_params.get(k, orig_space[k][0])
        spec_space[k] = get_symmetric_space(orig_space[k], chosen, is_float=False)
        
    # Floats (+/- 15%)
    float_keys = ['lr', 'weight_decay', 'dropout']
    for k in float_keys:
        chosen = best_foundation_params.get(k, orig_space[k][0])
        spec_space[k] = get_symmetric_space(orig_space[k], chosen, is_float=True)
        
    return spec_space

def load_data(directory: str, feature_cols: list):
    parquet_files = sorted(list(Path(directory).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No labelled data in {directory}")

    dfs = []
    for i, pf in enumerate(parquet_files):
        df_i = pl.read_parquet(pf, columns=feature_cols + ['target', 'island_id'])
        # Protocol Island Split: ensure global uniqueness across files
        df_i = df_i.with_columns(pl.col("island_id") + (i * 10000))
        dfs.append(df_i)
    
    df = pl.concat(dfs)
    return df

class SpecialistObjective:
    def __init__(self, config, spec_space, X_train, y_train, island_t, X_val, y_val, island_v, class_weights, DEVICE):
        self.config = config
        self.spec_space = spec_space
        self.X_train = X_train
        self.y_train = y_train
        self.island_t = island_t
        self.X_val = X_val
        self.y_val = y_val
        self.island_v = island_v
        self.class_weights = class_weights
        self.DEVICE = DEVICE
        
        self.feature_cols_len = X_train.shape[1]
        self.epochs = config['optimization']['search_space']['epochs']
        self.patience = config['optimization']['search_space']['early_stopping_patience']

    def __call__(self, trial: optuna.Trial):
        # 1. Suggest parameters from restricted 3-option categorical grid
        batch_size = trial.suggest_categorical("batch_size", self.spec_space["batch_size"])
        seq_len = trial.suggest_categorical("seq_len", self.spec_space["seq_len"])
        lr = trial.suggest_categorical("lr", self.spec_space["lr"])
        weight_decay = trial.suggest_categorical("weight_decay", self.spec_space["weight_decay"])
        dropout = trial.suggest_categorical("dropout", self.spec_space["dropout"])
        
        tcn_channels = trial.suggest_categorical("tcn_channels", self.spec_space["tcn_channels"])
        lstm_hidden = trial.suggest_categorical("lstm_hidden", self.spec_space["lstm_hidden"])
        num_lstm_layers = trial.suggest_categorical("num_lstm_layers", self.spec_space["num_lstm_layers"])

        # 2. Datasets & Loaders
        train_dataset = SequenceDataset(self.X_train, self.y_train, self.island_t, seq_len)
        val_dataset = SequenceDataset(self.X_val, self.y_val, self.island_v, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

        # 3. Model
        model = Hybrid_TCN_LSTM(
            num_features=self.feature_cols_len,
            seq_len=seq_len,
            tcn_channels=tcn_channels,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers,
            num_classes=3,
            dropout=dropout,
        ).to(self.DEVICE)
        
        # We start from scratch for specialist? Or transfer learning? 
        # The architecture can change (e.g. layers, channels). So we train from scratch
        # in the constrained neighborhood space to find the optimal specific weights.

        # 4. Criterion & Optimizer
        alpha = torch.tensor(self.class_weights, dtype=torch.float32).to(self.DEVICE)
        gamma = self.config['training']['specialization_weights'].get('gamma', 2.0)
        smoothing = self.config['training']['specialization_weights'].get('smoothing', 0.1)
        criterion = FocalLossWithSmoothing(alpha=alpha, gamma=gamma, smoothing=smoothing)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        amp_scaler = torch.amp.GradScaler('cuda')

        best_sniper_score = 0.0
        patience_counter = 0

        # 5. Training Loop
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.DEVICE), batch_y.to(self.DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs["logits"], batch_y)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                train_loss += loss.item()
                
            scheduler.step()

            # Validation Loop
            model.eval()
            all_preds, all_targets = [], []
            val_loss = 0.0
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.DEVICE), batch_y.to(self.DEVICE)
                        outputs = model(batch_X)
                        val_loss += criterion(outputs["logits"], batch_y).item()
                        preds = torch.argmax(outputs["logits"], dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(batch_y.cpu().numpy())

            f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            f1_per_cls = f1_score(all_targets, all_preds, average=None, zero_division=0)
            f1_dir = (f1_per_cls[0] + f1_per_cls[2]) / 2.0
            
            # --- SNIPER SCORE CALCULATION ---
            w_dir = self.config['optimization'].get('sniper_weights', {}).get('dir', 0.7)
            w_macro = self.config['optimization'].get('sniper_weights', {}).get('macro', 0.3)
            sniper_score = (w_dir * f1_dir) + (w_macro * f1_macro)
            
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"[Trial {trial.number}] E{epoch+1}/{self.epochs} | "
                    f"🏆 Sniper Score: {sniper_score:.4f} | F1_Dir: {f1_dir:.4f} | F1_Macro: {f1_macro:.4f} | "
                    f"[S/N/B]: [{f1_per_cls[0]:.2f}/{f1_per_cls[1]:.2f}/{f1_per_cls[2]:.2f}]"
                )

            trial.report(sniper_score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if sniper_score > best_sniper_score:
                best_sniper_score = sniper_score
                patience_counter = 0
                # Saving the model state within the trial temporarily
                trial.set_user_attr("best_model_state", {k: v.cpu() for k, v in model.state_dict().items()})
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        return best_sniper_score

def run_specialization():
    setup_logger("specialization_optuna", "")
    config = load_config()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load best params from Foundation
    foundation_params_path = Path("src/cloud/base_model/otimizacao/best_params.json")
    if not foundation_params_path.exists():
        logger.error(f"❌ Foundation best_params.json missing at {foundation_params_path}. Run base optimization first.")
        sys.exit(1)
        
    with open(foundation_params_path, 'r', encoding='utf-8') as f:
        best_foundation_params = json.load(f)
        
    logger.info(f"Loaded Foundation Params: {best_foundation_params}")
    
    # 2. Build Specialist Space
    spec_space = build_specialist_space(config, best_foundation_params)
    logger.info("📐 Specialist Search Space (Symmetric Delta Applied):")
    for k, v in spec_space.items():
        logger.info(f"  - {k}: {v}")
        
    # 3. Load Data
    if 'paths' not in config: config['paths'] = {'train_dir': 'AUTO', 'val_dir': 'AUTO'}
    config['paths']['train_dir'], config['paths']['val_dir'] = resolve_data_paths(config['paths'])
    
    # Specialized split is splits_specialized_labelled_... (as per split_dataset.py)
    spec_train_dir = Path(get_specialized_dir(config)) / "train"
    spec_val_dir   = Path(get_specialized_dir(config)) / "val"
    
    logger.info(f"📂 Loading Specialized Splits explicitly from: {spec_train_dir}")
    feature_cols = config['model']['feature_names']
    
    train_df = load_data(spec_train_dir, feature_cols)
    val_df   = load_data(spec_val_dir, feature_cols)
    
    X_train_raw = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train_raw = train_df.select('target').to_numpy().flatten().astype(np.int64)
    island_t    = train_df.select('island_id').to_numpy().flatten()
    X_val_raw   = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val_raw   = val_df.select('target').to_numpy().flatten().astype(np.int64)
    island_v    = val_df.select('island_id').to_numpy().flatten()
    
    # Normalization
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train_norm = scaler.transform(X_train_raw).astype(np.float32)
    X_val_norm   = scaler.transform(X_val_raw).astype(np.float32)
    
    scaler_path = Path(config['pipeline_paths']['scaler_specialized'])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    class_weights = config['training']['specialization_weights'].get('class_weights', [4.85, 0.38, 4.61])
    n_trials = config['optimization'].get('n_trials_specialist', 40)
    
    logger.info(f"🎯 Starting Specialist Optimization (Target: Sniper Score). Trials: {n_trials}")

    objective = SpecialistObjective(
        config=config, 
        spec_space=spec_space,
        X_train=X_train_norm, y_train=y_train_raw,
        island_t=island_t,
        X_val=X_val_norm, y_val=y_val_raw,
        island_v=island_v,
        class_weights=class_weights,
        DEVICE=DEVICE
    )
    
    study_name = "quantgod_specialist_v1"
    storage_name = "sqlite:///optuna_specialist.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"✅ Optimization finished. Best Sniper Score: {study.best_value:.4f}")
    logger.info(f"🏆 Best Hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
        
    # Save best parameters
    best_params_path = Path("src/cloud/base_model/otimizacao/best_params_specialist.json")
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    # Save the physical model of the best trial
    best_trial = study.best_trial
    if "best_model_state" in best_trial.user_attrs:
        model_output_path = Path(project_root) / config['pipeline_paths']['best_specialized_model']
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_trial.user_attrs["best_model_state"], model_output_path)
        logger.info(f"💾 Best Specialized Model Saved to: {model_output_path}")

if __name__ == "__main__":
    run_specialization()
