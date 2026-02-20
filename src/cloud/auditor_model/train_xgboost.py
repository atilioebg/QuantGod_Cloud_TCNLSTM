
"""
train_xgboost.py — XGBoost Auditor Model Training

Engineering Constraint #3: STRICT OOF WALK-FORWARD (zero data leakage).

Pipeline:
    1. Load the full chronological dataset (same labelled parquets used for TCN+LSTM).
    2. Split into K temporal folds (using TimeSeriesSplit).
    3. For each fold N:
       - Train Hybrid_TCN_LSTM on Folds 1..N-1 (standard gradient descent).
       - Generate Out-of-Fold (OOF) predictions on Fold N ONLY (no information leakage).
    4. Accumulate all OOF predictions across folds.
    5. Train XGBoost on [OOF_probs || OOF_meta_features] → Ground Truth.
    6. Evaluate XGBoost on a held-out final 10% (true test set, never used in either model).
"""

import xgboost as xgb
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import yaml
import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
import sys

project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.auditor_model.feature_engineering_meta import (
    extract_meta_features_batch, META_FEATURE_NAMES
)

log_dir = Path("logs/auditor")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"xgboost_training_{timestamp}.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X; self.y = y; self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx: idx + self.seq_len]),
                torch.tensor(self.y[idx + self.seq_len - 1], dtype=torch.long))


def load_configs():
    base_cfg_path = Path("src/cloud/base_model/configs/base_model_config.yaml")
    aud_cfg_path  = Path("src/cloud/auditor_model/configs/auditor_config.yaml")
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(aud_cfg_path) as f:
        aud_cfg = yaml.safe_load(f)
    return base_cfg, aud_cfg


def train_tcn_lstm_on_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    base_cfg: dict, aud_cfg: dict, device: torch.device
) -> Hybrid_TCN_LSTM:
    """
    Train Hybrid_TCN_LSTM on a single fold's training split.
    Returns the trained model for OOF prediction on the held-out fold.
    """
    seq_len     = base_cfg['training']['seq_len']
    class_weights = base_cfg['training']['class_weights']
    hp          = aud_cfg['base_model_hyperparameters']

    dataset     = SequenceDataset(X_train, y_train, seq_len)
    loader      = DataLoader(dataset, batch_size=hp['batch_size'], shuffle=True,
                             num_workers=4, pin_memory=True)

    model = Hybrid_TCN_LSTM(
        num_features=X_train.shape[1],
        seq_len=seq_len,
        tcn_channels=hp['tcn_channels'],
        lstm_hidden=hp['lstm_hidden'],
        num_lstm_layers=hp['num_lstm_layers'],
        num_classes=3,
        dropout=hp['dropout'],
    ).to(device)

    weights   = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'])
    amp_scaler = torch.amp.GradScaler('cuda')

    for epoch in range(hp['epochs']):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                out  = model(batch_X)
                loss = criterion(out["logits"], batch_y)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / len(loader)
        logger.info(f"  [TCN+LSTM Fold Train] Epoch {epoch+1}/{hp['epochs']} | Loss: {avg:.4f}")

    return model


def generate_oof_predictions(
    model: Hybrid_TCN_LSTM,
    X_fold: np.ndarray, y_fold: np.ndarray,
    base_cfg: dict, aud_cfg: dict,
    device: torch.device,
) -> tuple:
    """
    Generate OOF predictions on a single fold using the model trained on previous folds.
    Returns (meta_features: (N, 14), y_true: (N,)) for the fold.
    """
    seq_len = base_cfg['training']['seq_len']
    hp      = aud_cfg['base_model_hyperparameters']

    # Only samples where we have a full sequence are valid
    X_seqs  = np.array([X_fold[i: i + seq_len] for i in range(len(X_fold) - seq_len)])
    y_true  = y_fold[seq_len:]

    meta = extract_meta_features_batch(
        X_seqs, model, device, batch_size=hp.get('inference_batch_size', 512)
    )
    return meta, y_true


def run_walk_forward_oof(base_cfg: dict, aud_cfg: dict) -> tuple:
    """
    Engineering Constraint #3 — Strict OOF Walk-Forward:
    - TimeSeriesSplit divides the dataset into K temporal folds.
    - TCN+LSTM trains on folds 1..N-1, predicts on fold N only.
    - OOF predictions never overlap with the backprop data.
    - Accumulated OOF predictions → XGBoost training set.

    Returns:
        X_oof: (M, 14) — accumulated OOF meta-features
        y_oof: (M,)   — corresponding ground-truth labels
    """
    feature_cols = base_cfg['model']['feature_names']
    n_splits     = aud_cfg['walk_forward']['n_folds']
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading labelled data for walk-forward OOF generation...")
    labelled_dir = aud_cfg['paths']['labelled_dir']
    parquet_files = sorted(Path(labelled_dir).glob("*.parquet"))
    dfs = [pl.read_parquet(p, columns=feature_cols + ['target']) for p in parquet_files]
    df  = pl.concat(dfs)
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_raw = df.select('target').to_numpy().flatten().astype(np.int64)

    # Reserve final 10% as true held-out test (never touches OOF or XGBoost training)
    test_cutoff = int(len(X_raw) * 0.9)
    X_dev, y_dev = X_raw[:test_cutoff], y_raw[:test_cutoff]
    X_test, y_test = X_raw[test_cutoff:], y_raw[test_cutoff:]
    logger.info(f"Dev set: {len(X_dev):,} | True test (held-out): {len(X_test):,}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_meta   = []
    all_y_true = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
        logger.info(f"=== Walk-Forward Fold {fold_idx+1}/{n_splits} ===")
        logger.info(f"  Train: {len(train_idx):,} | Val (OOF): {len(val_idx):,}")

        X_train_fold, y_train_fold = X_dev[train_idx], y_dev[train_idx]
        X_val_fold,   y_val_fold   = X_dev[val_idx],   y_dev[val_idx]

        # Normalize: fit on fold's train only (zero leakage)
        scaler = StandardScaler()
        scaler.fit(X_train_fold)
        X_train_norm = scaler.transform(X_train_fold).astype(np.float32)
        X_val_norm   = scaler.transform(X_val_fold).astype(np.float32)

        # Train TCN+LSTM on this fold's training split
        model = train_tcn_lstm_on_fold(X_train_norm, y_train_fold, base_cfg, aud_cfg, device)

        # Generate OOF predictions on the val part (NEVER seen during backprop)
        meta, y_true = generate_oof_predictions(model, X_val_norm, y_val_fold,
                                                  base_cfg, aud_cfg, device)
        all_meta.append(meta)
        all_y_true.append(y_true)
        logger.info(f"  OOF predictions generated: {len(y_true):,} samples")

        # Cleanup GPU memory between folds
        del model
        torch.cuda.empty_cache()

    X_oof = np.vstack(all_meta)    # (M, 14)
    y_oof = np.concatenate(all_y_true)  # (M,)
    logger.info(f"Total OOF pool: {len(y_oof):,} samples with {X_oof.shape[1]} meta-features")
    return X_oof, y_oof, X_test, y_test, X_raw, y_raw


def train_xgboost(X_oof, y_oof, aud_cfg):
    """Train XGBoost auditor on OOF predictions with temporal CV."""
    xgb_params = aud_cfg['xgboost']
    logger.info(f"Training XGBoost on {len(y_oof):,} OOF samples...")
    logger.info(f"XGBoost params: {xgb_params}")

    model = xgb.XGBClassifier(
        n_estimators    = xgb_params.get('n_estimators', 500),
        max_depth       = xgb_params.get('max_depth', 6),
        learning_rate   = xgb_params.get('learning_rate', 0.05),
        subsample       = xgb_params.get('subsample', 0.8),
        colsample_bytree= xgb_params.get('colsample_bytree', 0.8),
        reg_alpha       = xgb_params.get('reg_alpha', 0.1),
        reg_lambda      = xgb_params.get('reg_lambda', 1.0),
        objective       = 'multi:softprob',
        num_class       = 3,
        eval_metric     = 'mlogloss',
        use_label_encoder=False,
        tree_method     = 'hist',
        device          = 'cuda' if torch.cuda.is_available() else 'cpu',
        feature_names   = META_FEATURE_NAMES,
        verbosity       = 1,
        seed            = 42,
    )

    # Walk-forward internal CV for early stopping
    tscv_internal = TimeSeriesSplit(n_splits=3)
    train_idx_final, val_idx_final = list(tscv_internal.split(X_oof))[-1]
    eval_set = [(X_oof[val_idx_final], y_oof[val_idx_final])]

    model.fit(
        X_oof, y_oof,
        eval_set=eval_set,
        verbose=50,
    )
    return model


def run_xgboost_training():
    base_cfg, aud_cfg = load_configs()

    # Step 1: Generate OOF predictions (strict walk-forward)
    X_oof, y_oof, X_test, y_test, X_raw_full, y_raw_full = run_walk_forward_oof(base_cfg, aud_cfg)

    # Step 2: Train XGBoost on OOF pool
    xgb_model = train_xgboost(X_oof, y_oof, aud_cfg)

    # Step 3: Evaluate on held-out test set
    # For test set: use the FINAL trained model (on full training data) to generate probs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = base_cfg['training']['seq_len']
    hp = aud_cfg['base_model_hyperparameters']
    feature_cols = base_cfg['model']['feature_names']

    logger.info("Generating test meta-features using final TCN+LSTM...")
    test_cutoff   = int(len(X_raw_full) * 0.9)
    train_cutoff  = int(test_cutoff * 0.8)
    scaler_final  = StandardScaler()
    scaler_final.fit(X_raw_full[:train_cutoff].astype(np.float32))
    X_test_norm   = scaler_final.transform(X_test.astype(np.float32)).astype(np.float32)
    X_train_norm  = scaler_final.transform(X_raw_full[:test_cutoff].astype(np.float32)).astype(np.float32)

    final_model = train_tcn_lstm_on_fold(
        X_train_norm[:train_cutoff], y_raw_full[:train_cutoff],
        base_cfg, aud_cfg, device
    )
    X_test_seqs = np.array([X_test_norm[i: i + seq_len] for i in range(len(X_test_norm) - seq_len)])
    y_test_valid = y_test[seq_len:]
    X_test_meta  = extract_meta_features_batch(X_test_seqs, final_model, device, batch_size=512)

    y_pred = xgb_model.predict(X_test_meta)
    f1_macro = f1_score(y_test_valid, y_pred, average='macro', zero_division=0)
    f1_per   = f1_score(y_test_valid, y_pred, average=None,    zero_division=0)
    logger.info(f"=== XGBoost Held-Out Test Results ===")
    logger.info(f"F1 Macro: {f1_macro:.4f} | [SELL/NEU/BUY]: {f1_per}")
    logger.info(f"\n{classification_report(y_test_valid, y_pred, target_names=['SELL','NEUTRAL','BUY'], zero_division=0)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test_valid, y_pred)}")

    # Feature importance
    importance = xgb_model.feature_importances_
    imp_ranked = sorted(zip(META_FEATURE_NAMES, importance), key=lambda x: -x[1])
    logger.info("XGBoost Feature Importance (top 14):")
    for name, imp in imp_ranked:
        logger.info(f"  {name:20s}: {imp:.4f}")

    # Step 4: Save artifacts
    model_out = Path(aud_cfg['paths']['xgb_model_output'])
    model_out.parent.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(model_out))
    logger.info(f"XGBoost model saved: {model_out}")

    with open(aud_cfg['paths']['scaler_output'], 'wb') as f:
        pickle.dump(scaler_final, f)

    del final_model
    torch.cuda.empty_cache()
    logger.info("XGBoost auditor training complete.")


if __name__ == "__main__":
    run_xgboost_training()
