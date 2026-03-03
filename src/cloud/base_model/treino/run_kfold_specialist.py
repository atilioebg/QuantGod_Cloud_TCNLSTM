"""
run_kfold_specialist.py — Fase 2: K-Fold Out-of-Fold Specialist

Implements the Purged Blocked K-Fold strategy for the Specialist TCN-LSTM model.
This produces OOF (Out-of-Fold) predictions covering 100% of the Foundation
Validation set — feeding the Auditor XGBoost ~5x more data than the legacy Holdout.

Design Principles (Anti-Leakage):
  1. Blocked K-Fold:  Splits are chronological blocks, not random.
  2. Purge Gap:       `purge_minutes` rows are removed from the training boundary
                      adjacent to the test block. This ensures no feature (e.g. a 
                      15-min rolling window) computed DURING training can "see"
                      the test block's labels.
  3. Per-Fold Scaler: Each fold fits a fresh StandardScaler only on ITS training data.
                      Never a global scaler (would leak test distribution into train).
  4. Logit Output:    Saves raw softmax probabilities (3 columns) — not just argmax class.
                      The Auditor needs calibrated confidence, not just direction.

Output:
  data/auditor/oof_predictions/fold_{k}.parquet   (per-fold logits + targets)
  data/auditor/oof_predictions/full_oof.parquet   (concatenated, chronological)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import polars as pl
import numpy as np
import pandas as pd
import yaml
import logging
import pickle
import json
import sys
import gc
from pathlib import Path
from datetime import datetime

project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing
from src.cloud.base_model.utils.logging_utils import setup_logger, upload_audit_to_drive
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """Island-aware sliding window dataset. Sequences that cross island boundaries
    are excluded to prevent the model from learning artificial cross-gap patterns.
    Mirrors run_specialization.py exactly."""
    def __init__(self, X: np.ndarray, y: np.ndarray, island_ids: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

        # Pre-calculate valid indices: window [idx, idx+seq_len) must stay within one island.
        max_idx = len(X) - seq_len
        if max_idx < 0:
            self.valid_indices = np.array([], dtype=np.int64)
        else:
            # island_ids[idx] == island_ids[idx + seq_len - 1]  ↔  no island crossing
            self.valid_indices = np.where(
                island_ids[:max_idx + 1] == island_ids[seq_len - 1:]
            )[0].astype(np.int64)

        logger.info(
            f"SequenceDataset: {len(self.valid_indices)}/{max_idx + 1 if max_idx >= 0 else 0} "
            f"valid sequences (Lookback protection active)."
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x_seq   = self.X[real_idx: real_idx + self.seq_len]
        y_label = self.y[real_idx + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.long)


# ── Config ───────────────────────────────────────────────────────────────────
def load_config() -> dict:
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ── Purged Blocked K-Fold ─────────────────────────────────────────────────────
def blocked_purged_kfold_indices(n: int, n_splits: int, purge_bars: int):
    """
    Generates (train_indices, test_indices) for Blocked Purged K-Fold.

    For a dataset of size n with n_splits folds:
    - Divides into n_splits equal-size contiguous blocks.
    - For each fold k, the TEST block is block k.
    - The TRAIN blocks are ALL OTHER blocks, with a purge_bars gap removed
      from the boundaries adjacent to the test block.

    The purge ensures that no rolling-window feature (e.g. a 15-min spread Z-Score
    computed at minute t) can incorporate information from the test block.

    Args:
        n:           Total number of samples.
        n_splits:    Number of folds (K).
        purge_bars:  Number of rows to exclude at adjacency boundaries.
                     Set to: purge_minutes / resample_freq_minutes.

    Yields:
        (train_idx, test_idx): numpy arrays of integer indices.
    """
    block_size = n // n_splits
    for fold_k in range(n_splits):
        # Test block: contiguous block of rows
        test_start = fold_k * block_size
        test_end   = test_start + block_size if fold_k < n_splits - 1 else n
        test_idx   = np.arange(test_start, test_end)

        # Train blocks: all rows outside test block, with purge boundary removed
        train_idx_parts = []
        for other_k in range(n_splits):
            if other_k == fold_k:
                continue
            other_start = other_k * block_size
            other_end   = other_start + block_size if other_k < n_splits - 1 else n

            # Apply purge: remove rows adjacent to the test block boundary
            # If block is BEFORE test block → remove its last `purge_bars` rows
            # If block is AFTER  test block → remove its first `purge_bars` rows
            if other_end <= test_start:
                purged_end   = max(other_start, other_end - purge_bars)
                train_idx_parts.append(np.arange(other_start, purged_end))
            elif other_start >= test_end:
                purged_start = min(other_end, other_start + purge_bars)
                train_idx_parts.append(np.arange(purged_start, other_end))
            else:
                # Overlapping (should not happen in blocked K-Fold)
                logger.warning(f"[kfold] Unexpected block overlap at fold {fold_k}, skipping block {other_k}")

        if not train_idx_parts:
            logger.error(f"[kfold] Fold {fold_k}: empty training set after purge!")
            continue

        train_idx = np.concatenate(train_idx_parts)
        yield train_idx, test_idx


# ── Single-Fold Training ──────────────────────────────────────────────────────
def train_specialist_fold(
    X_train_norm: np.ndarray,
    y_train: np.ndarray,
    island_train: np.ndarray,
    X_val_norm: np.ndarray,
    y_val: np.ndarray,
    island_val: np.ndarray,
    config: dict,
    best_params: dict,
    class_weights: list,
    DEVICE: torch.device,
    fold_k: int,
) -> torch.nn.Module:
    """
    Trains one clone of the Specialist TCN-LSTM model for a single K-Fold.

    Args:
        X_train_norm:   Normalized feature array for training.
        y_train:        Integer label array for training.
        X_val_norm:     Normalized feature array for validation (used only for early stopping).
        y_val:          Integer label array for validation.
        config:         Master config dict.
        best_params:    best_params.json from Foundation Optuna (architecture definition).
        class_weights:  List of 3 floats for FocalLoss alpha.
        DEVICE:         torch.device (cuda or cpu).
        fold_k:         Fold index (for logging).

    Returns:
        Trained model at best validation Sniper Score.
    """
    seq_len      = best_params['seq_len']
    tcn_channels = best_params['tcn_channels']
    lstm_hidden  = best_params['lstm_hidden']
    num_lstm_layers = best_params['num_lstm_layers']
    dropout      = best_params['dropout']
    lr           = best_params['lr']
    weight_decay = best_params['weight_decay']
    batch_size   = best_params.get('batch_size', 512)

    epochs   = config['pre_processing']['kfold']['specialist_epochs']
    patience = config['optimization']['search_space']['early_stopping_patience']

    num_features = X_train_norm.shape[1]

    # Datasets & Loaders
    train_ds = SequenceDataset(X_train_norm, y_train, island_train, seq_len)
    val_ds   = SequenceDataset(X_val_norm,   y_val,   island_val,   seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Fold {fold_k}: dataset too small after purge (train={len(train_ds)}, val={len(val_ds)})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Model (fresh clone, same architecture as Foundation winner)
    model = Hybrid_TCN_LSTM(
        num_features=num_features,
        seq_len=seq_len,
        tcn_channels=tcn_channels,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_lstm_layers,
        num_classes=3,
        dropout=dropout,
    )
    
    # ── Safe Warm-start Injection ─────────────────────────────────────────────
    # Load Foundation model weights if shapes match
    base_model_path = Path(config['pipeline_paths']['best_tcn_lstm_model'])
    if base_model_path.exists():
        try:
            state_dict = torch.load(base_model_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Check shape compatibility (especially input features)
            incompatible = False
            for name, param in model.state_dict().items():
                if name in state_dict and param.shape != state_dict[name].shape:
                    incompatible = True
                    break
            
            if incompatible:
                logger.warning(
                    f"⚠️ [Fold {fold_k}] WARM-START ABORTED: Checkpoint features shape mismatch. "
                    f"Expected input features: {num_features}. "
                    f"Falling back to COLD-START (Random Initialization) for this clone."
                )
            else:
                model.load_state_dict(state_dict)
                logger.info(f"🔥 [Fold {fold_k}] WARM-START SUCCESS: Base weights initialized.")
        except Exception as e:
            logger.error(f"⚠️ [Fold {fold_k}] WARM-START FAILURE: {e}. Falling back to COLD-START.")
    
    model = model.to(DEVICE)

    # Loss & Optimizer
    alpha    = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    gamma    = config['training']['specialization_weights'].get('gamma', 2.0)
    smoothing = config['training']['specialization_weights'].get('smoothing', 0.1)
    criterion = FocalLossWithSmoothing(alpha=alpha, gamma=gamma, smoothing=smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    w_dir   = config['optimization'].get('sniper_weights', {}).get('dir', 0.7)
    w_macro = config['optimization'].get('sniper_weights', {}).get('macro', 0.3)

    best_score = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            if amp_scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs["logits"], batch_y)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs["logits"], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            ctx = torch.amp.autocast('cuda') if DEVICE.type == 'cuda' else torch.no_grad()
            with ctx:
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    outputs = model(batch_X)
                    preds   = torch.argmax(outputs["logits"], dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.numpy())

        f1_macro  = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_per    = f1_score(all_targets, all_preds, average=None, zero_division=0)
        f1_dir    = (f1_per[0] + f1_per[2]) / 2.0 if len(f1_per) >= 3 else 0.0
        sniper    = (w_dir * f1_dir) + (w_macro * f1_macro)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"[Fold {fold_k}] E{epoch+1}/{epochs} | Sniper: {sniper:.4f} | "
                f"F1_Dir: {f1_dir:.4f} | F1_Macro: {f1_macro:.4f}"
            )

        if sniper > best_score:
            best_score   = sniper
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[Fold {fold_k}] Early stop at epoch {epoch+1}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── Inference on OOF fold ─────────────────────────────────────────────────────
def run_inference(model: nn.Module, X_norm: np.ndarray, y: np.ndarray,
                  island_ids: np.ndarray, seq_len: int, batch_size: int, DEVICE: torch.device):
    """
    Runs inference on the OOF test block and returns softmax probabilities + true targets.

    Returns:
        probs   (N, 3): Softmax probabilities [P(SELL), P(NEU), P(BUY)]
        targets (N,):   Ground-truth labels (aligned with SequenceDataset offset)
    """
    dataset = SequenceDataset(X_norm, y, island_ids, seq_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        ctx = torch.amp.autocast('cuda') if DEVICE.type == 'cuda' else torch.no_grad()
        with ctx:
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(DEVICE)
                outputs = model(batch_X)
                probs   = torch.softmax(outputs["logits"], dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(batch_y.numpy())

    return np.vstack(all_probs), np.concatenate(all_targets)


# ── Main K-Fold Loop ──────────────────────────────────────────────────────────
def run_kfold_specialist():
    setup_logger("kfold_specialist", "")
    config = load_config()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {DEVICE}")

    # ── 1. Load K-Fold config ─────────────────────────────────────────────────
    kfold_cfg  = config['pre_processing']['kfold']
    n_splits   = kfold_cfg.get('n_splits', 5)
    purge_min  = kfold_cfg.get('purge_minutes', 15)
    oof_dir    = Path(kfold_cfg.get('oof_output_dir', 'data/auditor/oof_predictions'))
    oof_dir.mkdir(parents=True, exist_ok=True)

    # purge_bars = purge_minutes / resample_freq_minutes
    resample_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
    resample_min  = int(resample_freq.replace('min', '').replace('T', ''))
    purge_bars    = max(1, purge_min // resample_min)
    logger.info(f"📐 K-Fold Setup: {n_splits} folds, purge={purge_min}min ({purge_bars} bars), resample={resample_freq}")

    # ── 2. Load Foundation best params ───────────────────────────────────────
    base_params_path = Path("src/cloud/base_model/otimizacao/best_params.json")
    if not base_params_path.exists():
        logger.error("❌ best_params.json not found. Run Foundation Optuna first.")
        sys.exit(1)
    with open(base_params_path, 'r') as f:
        best_params = json.load(f)
    logger.info(f"Loaded Foundation Arch: seq_len={best_params['seq_len']}, tcn={best_params['tcn_channels']}, lstm={best_params['lstm_hidden']}×{best_params['num_lstm_layers']}")

    # ── 3. Load Foundation Validation data ───────────────────────────────────
    # Source: splits_{labelled_name}/val — the 30% hold-out from Foundation
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    base_labelled_name = f"labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    foundation_val_dir = Path(f"data/L2/splits_{base_labelled_name}/val")

    if not foundation_val_dir.exists():
        logger.error(f"❌ Foundation Val not found: {foundation_val_dir}. Run split_dataset.py first.")
        sys.exit(1)

    feature_cols = config['model']['feature_names']
    parquet_files = sorted(list(foundation_val_dir.glob("*.parquet")))
    logger.info(f"📂 Foundation Val: {len(parquet_files)} files in {foundation_val_dir}")

    dfs = []
    for i, pf in enumerate(parquet_files):
        df_i = pl.read_parquet(pf, columns=feature_cols + ['target', 'island_id'])
        # Offset island_id per file to guarantee global uniqueness across the concatenated array
        df_i = df_i.with_columns(pl.col('island_id') + (i * 10000))
        dfs.append(df_i)
    df_val = pl.concat(dfs)

    X_raw      = df_val.select(feature_cols).to_numpy().astype(np.float32)
    y_raw      = df_val.select('target').to_numpy().flatten().astype(np.int64)
    island_raw = df_val.select('island_id').to_numpy().flatten()
    n_total    = len(X_raw)
    logger.info(f"📊 Foundation Val: {n_total:,} rows | SELL={np.sum(y_raw==0):,} NEU={np.sum(y_raw==1):,} BUY={np.sum(y_raw==2):,}")

    # Label balance warning
    neutral_pct = np.sum(y_raw == 1) / n_total
    if neutral_pct > 0.95:
        logger.warning(
            f"⚠️  NEUTRAL = {neutral_pct:.1%} — extremamente desbalanceado! "
            f"Considere aumentar os thresholds (sell/buy) no master_config.yaml. "
            f"O Especialista terá poucos sinais de volatilidade para aprender."
        )

    class_weights = config['training']['specialization_weights'].get('class_weights', [4.85, 0.38, 4.61])
    batch_size    = best_params.get('batch_size', 512)
    seq_len       = best_params['seq_len']

    # ── 4. K-Fold Loop ────────────────────────────────────────────────────────
    fold_results = []   # list of (test_original_idx, probs, targets)

    for fold_k, (train_idx, test_idx) in enumerate(
        blocked_purged_kfold_indices(n_total, n_splits, purge_bars)
    ):
        logger.info("=" * 60)
        logger.info(f"🔁 FOLD {fold_k + 1}/{n_splits} | Train: {len(train_idx):,} rows | Test: {len(test_idx):,} rows")

        # Temporal Leakage Guard (logged for test_no_temporal_leakage)
        if len(train_idx) > 0 and len(test_idx) > 0:
            test_min = test_idx.min()
            test_max = test_idx.max()

            left_train = train_idx[train_idx < test_min]
            right_train = train_idx[train_idx > test_max]

            # Print gaps for validation
            if len(left_train) > 0:
                gap_left = test_min - left_train.max() - 1
                if gap_left < purge_bars:
                    logger.warning(f"[Fold {fold_k}] ⚠️ Left Gap VIOLATION: {gap_left} < {purge_bars} bars!")
                else:
                    logger.info(f"[Fold {fold_k}] ✅ Left Purge Gap OK: {gap_left} bars ({gap_left * resample_min} min)")
                    
            if len(right_train) > 0:
                gap_right = right_train.min() - test_max - 1
                if gap_right < purge_bars:
                    logger.warning(f"[Fold {fold_k}] ⚠️ Right Gap VIOLATION: {gap_right} < {purge_bars} bars!")
                else:
                    logger.info(f"[Fold {fold_k}] ✅ Right Purge Gap OK: {gap_right} bars ({gap_right * resample_min} min)")

        X_train_raw  = X_raw[train_idx]
        y_train      = y_raw[train_idx]
        island_train = island_raw[train_idx]
        X_test_raw   = X_raw[test_idx]
        y_test       = y_raw[test_idx]
        island_test  = island_raw[test_idx]

        # ── Per-Fold Scaler (NEVER global) ────────────────────────────────────
        # Anti-Leakage: scaler is fit ONLY on fold's training data.
        # Using test stats in normalization would leak distribution into training.
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test_norm  = scaler.transform(X_test_raw).astype(np.float32)

        # Save fold scaler (needed if Auditor wants to replay inference in production)
        scaler_path = oof_dir / f"scaler_fold_{fold_k}.pkl"
        with open(scaler_path, 'wb') as sf:
            pickle.dump(scaler, sf)
        logger.info(f"[Fold {fold_k}] Scaler saved: {scaler_path.name}")

        # ── Train clone ───────────────────────────────────────────────────────
        model = train_specialist_fold(
            X_train_norm=X_train_norm,
            y_train=y_train,
            island_train=island_train,
            X_val_norm=X_test_norm,   # val used only for early stopping within fold
            y_val=y_test,
            island_val=island_test,
            config=config,
            best_params=best_params,
            class_weights=class_weights,
            DEVICE=DEVICE,
            fold_k=fold_k,
        )

        # ── OOF Inference (raw logits on unseen test block) ───────────────────
        probs, targets = run_inference(model, X_test_norm, y_test, island_test, seq_len, batch_size, DEVICE)
        pred_classes   = np.argmax(probs, axis=1)

        # Fold metrics
        f1_m = f1_score(targets, pred_classes, average='macro', zero_division=0)
        f1_c = f1_score(targets, pred_classes, average=None, zero_division=0)
        f1_d = (f1_c[0] + f1_c[2]) / 2.0 if len(f1_c) >= 3 else 0.0
        logger.info(f"[Fold {fold_k}] OOF Sniper: {0.7*f1_d + 0.3*f1_m:.4f} | Dir: {f1_d:.4f} | Macro: {f1_m:.4f}")

        # Store (using test_idx to allow global reordering at the end)
        fold_results.append({
            "test_idx":  test_idx,
            "probs":     probs,
            "targets":   targets,
        })

        # Save per-fold parquet (for debugging / partial resume)
        fold_df = pd.DataFrame({
            "original_row_idx": test_idx[:len(targets)],
            "spec_prob_sell":   probs[:, 0],
            "spec_prob_neu":    probs[:, 1],
            "spec_prob_buy":    probs[:, 2],
            "spec_pred_class":  pred_classes,
            "true_target":      targets,
            "fold":             fold_k,
        })
        fold_path = oof_dir / f"fold_{fold_k}.parquet"
        pl.DataFrame(fold_df).write_parquet(fold_path)
        logger.info(f"[Fold {fold_k}] Saved: {fold_path.name} ({len(fold_df):,} rows)")

        # Free GPU after each fold
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 5. Assemble full_oof.parquet ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("📦 Assembling full_oof.parquet (chronological order)...")

    # Concatenate and sort by original_row_idx to restore chronological order
    all_fold_dfs = [pl.read_parquet(oof_dir / f"fold_{k}.parquet") for k in range(n_splits)]
    full_oof_df  = pl.concat(all_fold_dfs).sort("original_row_idx")

    full_oof_path = oof_dir / "full_oof.parquet"
    full_oof_df.write_parquet(full_oof_path)

    # Final stats
    n_oof = len(full_oof_df)
    coverage = n_oof / n_total
    logger.info(f"✅ full_oof.parquet: {n_oof:,} rows | Coverage: {coverage:.1%} of Foundation Val")
    logger.info(f"   Expected: ~{n_total - n_splits * purge_bars:,} (after purge removal)")

    # Validate no duplicate original_row_idx
    n_unique = full_oof_df['original_row_idx'].n_unique()
    if n_unique < n_oof:
        logger.warning(f"⚠️  {n_oof - n_unique} duplicate rows in full_oof.parquet!")
    else:
        logger.info("✅ No duplicate rows in full_oof.parquet")

    # ── 6. Integrated Security QA ─────────────────────────────────────────────
    logger.info("🧪 Running Automated Specialist Security QA (pytest)...")
    try:
        import subprocess
        import os
        qa_log_path = oof_dir / "kfold_security_QA.log"
        with open(qa_log_path, 'w', encoding='utf-8') as qa_file:
            subprocess.run(
                ["pytest", "tests/kfold/test_kfold_security.py", "-v"],
                stdout=qa_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False
            )
        logger.info(f"✅ Specialist QA Report saved to {qa_log_path}")
    except Exception as e:
        logger.error(f"⚠️ Specialist QA Report generation failed: {e}")

    logger.info("🏁 K-Fold Specialist finished. Auditor fusion ready.")


if __name__ == "__main__":
    run_kfold_specialist()
    # Audit Logs → Drive  (PROJETOS/AUDITORIA/KFOLD_SPECIALIST)
    upload_audit_to_drive(
        local_dirs=["logs/kfold_specialist"],
        stage_name="KFOLD_SPECIALIST",
    )
