
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

# Project root on sys.path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from sklearn.preprocessing import StandardScaler

from src.cloud.base_model.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Memory-efficient demand-based sequence generator. No pre-allocation."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_label = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.long)


def load_config():
    # Training config (paths, seq_len overrides)
    training_cfg_path = Path("src/cloud/base_model/treino/training_config.yaml")
    with open(training_cfg_path, 'r') as f:
        train_cfg = yaml.safe_load(f)

    # Base model config: single source of truth for class_weights, features, etc.
    base_cfg_path = Path("src/cloud/base_model/configs/base_model_config.yaml")
    with open(base_cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # ── Check for optimized best_params.json ──────────────────────────────────
    best_params_path = Path("src/cloud/base_model/otimizacao/best_params.json")
    if best_params_path.exists():
        try:
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            
            # Update sequence length if it was optimized
            if 'seq_len' in best_params:
                train_cfg['hyperparameters']['seq_len'] = best_params['seq_len']
            
            # Map optimization keys to training keys (handling potential mismatches)
            opt_keys = ['lr', 'batch_size', 'dropout', 'tcn_channels', 'lstm_hidden', 'num_lstm_layers']
            for k in opt_keys:
                if k in best_params:
                    train_cfg['hyperparameters'][k] = best_params[k]
            
            logger.info(f"✨ OPTIMIZATION: Overriding hyperparameters from {best_params_path.name}")
            logger.debug(f"Optimized params: {best_params}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load best_params.json: {e}. Using YAML defaults.")

    return train_cfg, base_cfg


def load_data(directory: str, feature_cols: list):
    parquet_files = sorted(list(Path(directory).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No labelled data in {directory}")

    dfs = [pl.read_parquet(pf, columns=feature_cols + ['target']) for pf in parquet_files]
    df = pl.concat(dfs)
    logger.info(f"Loaded {len(df):,} rows from {len(parquet_files)} files in {directory}")
    return df


def run_training():
    # ── Load Config & Suffix Extraction ────────────────────────────────────────
    # 1. Load initial YAMLs
    training_cfg_path = Path("src/cloud/base_model/treino/training_config.yaml")
    with open(training_cfg_path, 'r') as f:
        train_cfg = yaml.safe_load(f)

    # Extract suffix for logging
    suffix = ""
    target_path = Path(train_cfg['paths']['train_dir'])
    import re
    match = re.search(r"(_SELL_.*)$", str(target_path.parent))
    if match:
        suffix = match.group(1)
        
    # 2. Setup Logger early to catch all initialization messages
    setup_logger("training", suffix)
    
    # 3. Load full config (with potential optimization overrides)
    train_cfg, base_cfg = load_config()

    # ── Config values ──────────────────────────────────────────────────────────
    feature_cols   = base_cfg['model']['feature_names']
    class_weights  = base_cfg['training']['class_weights']
    seq_len        = train_cfg['hyperparameters'].get('seq_len', base_cfg['training']['seq_len'])
    epochs         = train_cfg['hyperparameters'].get('epochs', base_cfg['training']['epochs'])
    patience       = base_cfg['training']['early_stopping_patience']
    clip_norm      = base_cfg['training'].get('gradient_clip_norm', 1.0)
    lr             = train_cfg['hyperparameters']['lr']
    batch_size     = train_cfg['hyperparameters']['batch_size']
    tcn_channels   = train_cfg['hyperparameters'].get('tcn_channels', 64)
    lstm_hidden    = train_cfg['hyperparameters'].get('lstm_hidden', 256)
    num_lstm_layers= train_cfg['hyperparameters'].get('num_lstm_layers', 2)
    dropout        = train_cfg['hyperparameters'].get('dropout', 0.3)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Training config: seq_len={seq_len}, epochs={epochs}, lr={lr}, batch={batch_size}")
    logger.info(f"Model Arch: TCN={tcn_channels}, LSTM={lstm_hidden}, Layers={num_lstm_layers}")
    logger.info(f"Class weights (SELL/NEUTRAL/BUY): {class_weights}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_df = load_data(train_cfg['paths']['train_dir'], feature_cols)
    val_df   = load_data(train_cfg['paths']['val_dir'], feature_cols)

    X_train_raw = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train_raw = train_df.select('target').to_numpy().flatten().astype(np.int64)
    
    X_val_raw   = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val_raw   = val_df.select('target').to_numpy().flatten().astype(np.int64)

    logger.info(f"Split sizes: train={len(X_train_raw):,} | val={len(X_val_raw):,}")

    # ── Normalization: fit on train only ───────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train_norm = scaler.transform(X_train_raw).astype(np.float32)
    X_val_norm   = scaler.transform(X_val_raw).astype(np.float32)

    scaler_path = Path(train_cfg['paths'].get('scaler_output', 'data/models/scaler_finetuning.pkl'))
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved: {scaler_path}")

    # ── Datasets and Loaders ───────────────────────────────────────────────────
    train_dataset = SequenceDataset(X_train_norm, y_train_raw, seq_len)
    val_dataset   = SequenceDataset(X_val_norm, y_val_raw, seq_len)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = Hybrid_TCN_LSTM(
        num_features=len(feature_cols),
        seq_len=seq_len,
        tcn_channels=tcn_channels,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_lstm_layers,
        num_classes=3,
        dropout=dropout,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: Hybrid_TCN_LSTM | Parameters: {total_params:,}")

    # ── Loss: weight loaded from centralized config ────────────────────────────
    weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    logger.info(f"CrossEntropyLoss weights: {class_weights}")

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_scaler = torch.amp.GradScaler('cuda')

    # ── Training Loop ──────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    patience_counter = 0
    model_output_path = Path(train_cfg['paths']['model_output'])
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0.0
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs["logits"], batch_y)
            amp_scaler.scale(loss).backward()
            # Gradient clipping stabilizes LSTM (prevents exploding gradients)
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                pct = (batch_idx + 1) / len(train_loader) * 100
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} ({pct:.1f}%) | Loss: {loss.item():.4f}")

        scheduler.step()

        # VALIDATE
        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs["logits"], batch_y).item()
                    preds = torch.argmax(outputs["logits"], dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())

        from sklearn.metrics import f1_score
        f1_macro   = f1_score(all_targets, all_preds, average='macro',    zero_division=0)
        f1_weighted= f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1_per_cls = f1_score(all_targets, all_preds, average=None,       zero_division=0)
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f} | "
            f"F1 [SELL/NEU/BUY]: [{f1_per_cls[0]:.3f}/{f1_per_cls[1]:.3f}/{f1_per_cls[2]:.3f}] | "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping + checkpoint
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), model_output_path)
            logger.info(f"✅ Best model saved (F1 Macro: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    logger.info(f"Training complete. Best Val F1 Macro: {best_val_f1:.4f}")
    logger.info(f"Model saved: {model_output_path}")


if __name__ == "__main__":
    run_training()
