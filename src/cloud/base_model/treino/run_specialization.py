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
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing, compute_alpha_from_labels
from sklearn.preprocessing import StandardScaler
from src.cloud.base_model.utils.logging_utils import setup_logger
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths

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
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
        
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ── For Specialization: strictly load MACRO champion params ────────────────
    params_filename = "best_params.json"
    best_params_path = Path("src/cloud/base_model/otimizacao") / params_filename
    if best_params_path.exists():
        try:
            with open(best_params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            opt_keys = ['lr', 'batch_size', 'dropout', 'tcn_channels', 'lstm_hidden',
                        'num_lstm_layers', 'seq_len']
            
            if 'training' not in config: config['training'] = {}
            if 'hyperparameters' not in config['training']: config['training']['hyperparameters'] = {}
                
            for k in opt_keys:
                if k in best_params:
                    config['training']['hyperparameters'][k] = best_params[k]
            logger.info(f"✨ [MACRO] Loading hyperparameters from {params_filename}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load {params_filename}: {e}. Using YAML defaults.")
    else:
        logger.warning(f"⚠️ {params_filename} not found. Using YAML defaults.")

    logger.info(f"📦 Specialization Model  → {config['pipeline_paths']['best_specialized_model']}")
    logger.info(f"📦 Specialization Scaler → {config['pipeline_paths']['scaler_specialized']}")
    return config


def load_data(directory: str, feature_cols: list):
    parquet_files = sorted(list(Path(directory).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No labelled data in {directory}")

    dfs = [pl.read_parquet(pf, columns=feature_cols + ['target']) for pf in parquet_files]
    df = pl.concat(dfs)
    logger.info(f"Loaded {len(df):,} rows from {len(parquet_files)} files in {directory}")
    return df

def run_specialization():
    # Setup Logger
    setup_logger("treino_specialization", "")
    
    # Load config
    config = load_config()

    # ── Config values ──────────────────────────────────────────────────────────
    feature_cols   = config['model']['feature_names']
    seq_len        = config['training']['hyperparameters'].get('seq_len', 24)
    epochs         = 10 # Enforced 10 epochs for specialization
    patience       = config['optimization']['search_space'].get('early_stopping_patience', 4)
    clip_norm      = 1.0
    
    # ── Transfer Learning: LR 10x smaller ──────────────────────────────────────
    base_lr        = config['training']['hyperparameters']['lr']
    lr             = base_lr * 0.1
    
    batch_size     = config['training']['hyperparameters']['batch_size']
    tcn_channels   = config['training']['hyperparameters'].get('tcn_channels', 64)
    lstm_hidden    = config['training']['hyperparameters'].get('lstm_hidden', 256)
    num_lstm_layers= config['training']['hyperparameters'].get('num_lstm_layers', 2)
    dropout        = config['training']['hyperparameters'].get('dropout', 0.3)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Specialization config: seq_len={seq_len}, epochs={epochs}, base_lr={base_lr:.6f}, spec_lr={lr:.6f}, batch={batch_size}")
    logger.info(f"Model Arch: TCN={tcn_channels}, LSTM={lstm_hidden}, Layers={num_lstm_layers}")

    # ── Data Loading Logic ─────────────────────────────────────────────────────
    # Standard: train = train_dir, val = val_dir
    # Specialization: Read directly from the newly generated 85/15 specialized splits
    # ── Resolve AUTO paths ─────────────────────────────────────────────────
    if 'paths' not in config: config['paths'] = {'train_dir': 'AUTO', 'val_dir': 'AUTO'}
    config['paths']['train_dir'], config['paths']['val_dir'] = resolve_data_paths(config['paths'])
    
    # Get standard base directory, then route to the specialized counterpart
    train_dir_path = Path(config['paths']['train_dir'])
    dataset_parent = train_dir_path.parent
    specialized_folder_name = f"specialized_{dataset_parent.name}"
    
    spec_train_dir = dataset_parent.parent / specialized_folder_name / "train"
    spec_val_dir   = dataset_parent.parent / specialized_folder_name / "val"

    logger.info(f"Specialization Train Set: {spec_train_dir}")
    logger.info(f"Specialization Val Set:   {spec_val_dir}")

    train_df = load_data(spec_train_dir, feature_cols)
    val_df   = load_data(spec_val_dir, feature_cols)

    X_train_raw = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train_raw = train_df.select('target').to_numpy().flatten().astype(np.int64)
    
    X_val_raw   = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val_raw   = val_df.select('target').to_numpy().flatten().astype(np.int64)

    logger.info(f"Split sizes: train={len(X_train_raw):,} | val={len(X_val_raw):,}")

    # ── Normalization: fit on specialization train ─────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train_norm = scaler.transform(X_train_raw).astype(np.float32)
    X_val_norm   = scaler.transform(X_val_raw).astype(np.float32)

    scaler_path = Path(config['pipeline_paths']['scaler_specialized'])
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
    
    # ── Load Base Model Weights (MACRO) ─────────────────────────────────────────
    base_model_weights = Path(project_root) / config['pipeline_paths']['best_tcn_lstm_model']
    if base_model_weights.exists():
        logger.info(f"🔄 Loading pre-trained base model weights from: {base_model_weights}")
        model.load_state_dict(torch.load(base_model_weights, map_location=DEVICE))
    else:
        logger.error(f"❌ Base model weights NOT FOUND at {base_model_weights}. Exiting.")
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: Hybrid_TCN_LSTM | Parameters: {total_params:,}")

    # ── Loss: dynamic alpha from training labels ───────────────────────────────
    spec_cfg = config['training'].get('specialization_weights', {})
    use_auto_class_weights = spec_cfg.get('use_auto_class_weights', True)
    
    if use_auto_class_weights:
        try:
            alpha = compute_alpha_from_labels(y_train_raw, num_classes=3, device=DEVICE)
            logger.info(f"FocalLoss alpha (AUTO computed from specialization labels): {alpha.cpu().tolist()}")
        except Exception as e:
            logger.error(f"compute_alpha_from_labels failed ({e}). Breaking process as requested by user.")
            raise
    else:
        class_weights = spec_cfg.get('class_weights', [1.0, 1.0, 1.0])
        alpha = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        logger.info(f"FocalLoss alpha (MANUAL from config): {class_weights}")

    gamma = spec_cfg.get('gamma', 2.0)
    smoothing = spec_cfg.get('smoothing', 0.1)
    criterion = FocalLossWithSmoothing(alpha=alpha, gamma=gamma, smoothing=smoothing)
    logger.info(f"Loss: FocalLossWithSmoothing | gamma={gamma} | smoothing={smoothing}")

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_scaler = torch.amp.GradScaler('cuda')

    # ── Training Loop ──────────────────────────────────────────────────────────
    best_val_metric = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    model_output_path = Path(project_root) / config['pipeline_paths']['best_specialized_model']
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
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
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

        val_loss_epoch = val_loss / len(val_loader)

        from sklearn.metrics import f1_score
        f1_macro   = f1_score(all_targets, all_preds, average='macro',    zero_division=0)
        f1_per_cls = f1_score(all_targets, all_preds, average=None,       zero_division=0)
        f1_dir     = (f1_per_cls[0] + f1_per_cls[2]) / 2
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"E{epoch+1}/{epochs} | "
            f"L: {train_loss/len(train_loader):.4f}/{val_loss_epoch:.4f} | "
            f"F1 M: {f1_macro:.4f} D: {f1_dir:.4f} | "
            f"[S/N/B]: [{f1_per_cls[0]:.2f}/{f1_per_cls[1]:.2f}/{f1_per_cls[2]:.2f}] | "
            f"LR: {current_lr:.2g}"
        )

        # Early stopping based on Configured Metric
        specialized_metric = config['optimization'].get('specialized_metric', 'f1_dir')
        current_metric_val = f1_macro if specialized_metric == 'f1_macro' else f1_dir

        if current_metric_val > best_val_metric:
            best_val_metric = current_metric_val
            patience_counter = 0
            torch.save(model.state_dict(), model_output_path)
            logger.info(f"🏆 Best Specialization model saved ({specialized_metric}: {best_val_metric:.8f})")
        else:
            patience_counter += 1

        # Checkpoint based on F1 Directional fallback info
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch

        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"⚠️ Early stopping triggered! {specialized_metric} did not improve for {patience} epochs.")
            break

    logger.info(f"Specialization complete. Best Val {specialized_metric}: {best_val_metric:.8f}")
    logger.info(f"Specialized Model saved: {model_output_path}")

if __name__ == "__main__":
    run_specialization()
