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
import subprocess
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import os

# Add project root to path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing
from src.cloud.base_model.utils.logging_utils import setup_logger, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import get_labelled_dir, get_drive_session_path, resolve_local_drive
from src.cloud.base_model.utils.dataset_utils import QuantGodLazyDataset
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────
# SequenceDataset agora é carregado via dataset_utils.py (QuantGodLazyDataset)


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
    train_dataset: QuantGodLazyDataset,
    val_dataset: QuantGodLazyDataset,
    config: dict,
    best_params: dict,
    class_weights: list,
    DEVICE: torch.device,
    fold_k: int,
) -> torch.nn.Module:
    """
    Trains one clone of the Specialist TCN-LSTM model for a single K-Fold.
    [v12.1] Streaming Version: Receives LazyDatasets already indexed.
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

    # Dataset Feature Stats fit via first batch or config (using the global search space n_features)
    num_features = len(config['model']['feature_names'])

    train_ds = train_dataset
    val_ds   = val_dataset

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Fold {fold_k}: dataset too small after purge (train={len(train_ds)}, val={len(val_ds)})")

    # Dynamic CPU Detection (v9.6 Robust)
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, ImportError, NotImplementedError):
        cpu_count = os.cpu_count() or 1
    
    # DataLoader workers: safe balance (max 4 or cpu_count/4)
    dl_workers = min(8, max(1, cpu_count // 4))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=dl_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=dl_workers)

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
    spec_cfg = config['training'].get('specialization_weights', {})
    use_dir = spec_cfg.get('use_best_f1_dir', False)
    mod_key = 'best_tcn_lstm_dir_model' if use_dir else 'best_tcn_lstm_model'
    
    base_dir = get_drive_session_path("MODELOS", config)
    warm_start_path = resolve_local_drive(Path(base_dir) / config['pipeline_paths'][mod_key])
    
    if not warm_start_path.exists():
        logger.error(f"❌ MODELO BASE AUSENTE CERIFIQUE")
        logger.error(f"   ↳ Caminho esperado: {warm_start_path}")
        sys.exit(1)

    logger.info(f"🔄 Warm-Starting from Foundation Checkpoint: {warm_start_path}")
    try:
        # weights_only=True is safer and removes FutureWarnings in newer PyTorch versions
        state_dict = torch.load(warm_start_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Check shape compatibility (especially input features)
        incompatible = False
        for name, param in model.state_dict().items():
            if name in state_dict and param.shape != state_dict[name].shape:
                incompatible = True
                break
        
        if incompatible:
            logger.error(f"❌ MODELO BASE INCOMPATIVEL CERIFIQUE")
            logger.error(f"   ↳ Erro: Shape mismatch no checkpoint.")
            sys.exit(1)
        else:
            model.load_state_dict(state_dict)
            logger.info(f"🔥 [Fold {fold_k}] WARM-START SUCCESS: Base weights initialized.")
    except Exception as e:
        logger.error(f"❌ ERRO AO CARREGAR MODELO BASE: {e}")
        sys.exit(1)
    
    model = model.to(DEVICE)

    # ── Focal Loss Params: Genetic Inheritance ──────────────────────────
    # Priority: best_params.json (Foundation Genetic) > master_config.yaml (Spec Override) > default 2.0/0.1
    spec_cfg = config['training'].get('specialization_weights', {})
    
    # Heritage logic: Foundation Params (Genetic) > Master Config Fallback
    gamma = best_params.get('base_loss_gamma')
    if gamma is None:
        gamma = spec_cfg.get('spec_gamma', 2.0)
        
    smoothing = best_params.get('base_loss_smoothing')
    if smoothing is None:
        smoothing = spec_cfg.get('spec_smoothing', 0.1)

    # Ensure class_weights (alpha) is a Tensor for FocalLossWithSmoothing
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    inh_bool = (gamma == best_params.get('base_loss_gamma'))
    inh_str = "YES" if inh_bool else "NO"
    logger.info(f"[LOSS] Specialist Loss: alpha={class_weights}, gamma={gamma:.2f}, "
                f"smoothing={smoothing:.2f} (Inherited: {inh_str})")
    
    criterion = FocalLossWithSmoothing(alpha=alpha_tensor, gamma=gamma, smoothing=smoothing)
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
def run_inference(model: nn.Module, test_dataset: QuantGodLazyDataset, DEVICE: torch.device):
    """
    Runs inference on the OOF test block and returns softmax probabilities + true targets.
    [v12.2] Streaming Inference: Reads from disk (Memory Safe).
    """
    loader  = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

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

    # valid_indices do test_dataset representam a posição absoluta no validation set
    return np.vstack(all_probs), np.concatenate(all_targets), test_dataset.global_indices


# ── Main K-Fold Loop ──────────────────────────────────────────────────────────
def run_kfold_specialist():
    setup_logger("kfold_specialist", "")
    config = load_config()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {DEVICE}")

    # ── 1. Load K-Fold config ─────────────────────────────────────────────────
    kfold_cfg  = config['pre_processing']['kfold']
    n_splits   = kfold_cfg.get('n_splits', 5)
    
    # ── Dinamismo Sniper: Purge automático com Margem de Segurança (N+1) ──────
    horizon_min = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    purge_min   = kfold_cfg.get('purge_minutes', 15)
    if purge_min <= 0:
        # Regra de Ouro: horizon + 1 minuto de margem para evitar vazamento em bordas de milissegundos
        purge_min = horizon_min + 1
        logger.info(f"🔄 Purge Dinamico Sniper ativado: {horizon_min}min (horizon) + 1min (seguranca) = {purge_min}min")
    
    base_dir = get_drive_session_path("MODELOS", config)
    oof_dir  = resolve_local_drive(Path(base_dir) / kfold_cfg.get('oof_output_dir', 'SPECIALIST'))
    oof_dir.mkdir(parents=True, exist_ok=True)

    # purge_bars = purge_minutes / resample_freq_minutes
    resample_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
    resample_min  = int(resample_freq.replace('min', '').replace('T', ''))
    purge_bars    = int(np.ceil(purge_min / resample_min))
    logger.info(f"📐 K-Fold Setup: {n_splits} folds, purge={purge_min}min ({purge_bars} bars), resample={resample_freq}")

    # ── 2. Load Foundation best params ───────────────────────────────────────
    spec_cfg = config['training'].get('specialization_weights', {})
    use_dir = spec_cfg.get('use_best_f1_dir', False)
    param_file = "best_dir_params.json" if use_dir else "best_params.json"
    
    base_params_path = Path("src/cloud/base_model/otimizacao") / param_file
    if not base_params_path.exists():
        logger.error(f"❌ {param_file} not found. Run Foundation Optuna first.")
        sys.exit(1)
    with open(base_params_path, 'r') as f:
        best_params = json.load(f)
    logger.info(f"Loaded Foundation Arch ({param_file}): seq_len={best_params['seq_len']}, tcn={best_params['tcn_channels']}, lstm={best_params['lstm_hidden']}×{best_params['num_lstm_layers']}")

    # ── [v12.5] REQUIREMENT CHECK (Fail Fast) ────────────────────────────────
    # Fonte: {labelled_dir}/val — o set de validação da Foundation
    foundation_val_dir = Path(get_labelled_dir(config)) / "val"
    if not foundation_val_dir.exists():
        logger.error(f"❌ Foundation Val not found: {foundation_val_dir}. Run split_dataset.py first.")
        sys.exit(1)

    feature_cols = config['model']['feature_names']
    parquet_files = sorted(list(foundation_val_dir.glob("*.parquet")))
    
    # ── [TEST MODE] LIMIT FILES ──────────
    test_limit = os.environ.get("QUANTGOD_TEST_LIMIT")
    if test_limit:
        n_limit = int(test_limit)
        logger.info(f"🧪 [TEST MODE] Limiting Specialist data load to {n_limit} files.")
        parquet_files = parquet_files[:n_limit]

    logger.info(f"📂 Foundation Val: {len(parquet_files)} files in {foundation_val_dir}")

    spec_cfg = config['training'].get('specialization_weights', {})
    use_dir = spec_cfg.get('use_best_f1_dir', False)
    mod_key = 'best_tcn_lstm_dir_model' if use_dir else 'best_tcn_lstm_model'
    
    base_dir_path = get_drive_session_path("MODELOS", config)
    warm_start_path = resolve_local_drive(Path(base_dir_path) / config['pipeline_paths'][mod_key])
    
    if not warm_start_path.exists():
        logger.error(f"❌ MODELO BASE AUSENTE: O Especialista precisa de um checkpoint da Foundation para o Warm-Start.")
        logger.error(f"   ↳ Certifique-se de que a fase Foundation Optuna gerou o arquivo: {warm_start_path.name}")
        logger.error(f"   ↳ Se o Trial da Foundation foi abortado (Cérebro Morto), tente aumentar n_trials ou ajustar os pesos.")
        sys.exit(1)

    # ── [v12.3] Big Data Map ─────────
    # Em vez de carregar tudo, criamos uma instância mestra do LazyDataset 
    # que contém o mapeamento de TODOS os arquivos do validation set.
    val_dataset_master = QuantGodLazyDataset(parquet_files, feature_cols, best_params['seq_len'], config, is_train=False)
    n_total = len(val_dataset_master)
    
    # Pre-carregamento dos labels reais p/ cálculo de alphas (Genetic Inheritance)
    # [v12.6] Optimização Polars: Scan de múltiplos arquivos ultra-rápido
    logger.info("📐 Pre-scanning labels for class balance stats...")
    y_raw = pl.scan_parquet(parquet_files).select('target').collect()['target'].to_numpy()
    
    logger.info(f"📊 Foundation Val: {n_total:,} rows (Mapped) | Labels: {len(y_raw):,}")

    # Label balance warning
    n_labels = len(y_raw)
    neutral_pct = (np.sum(y_raw == 1) / n_labels) if n_labels > 0 else 0
    if neutral_pct > 0.95:
        logger.warning(
            f"⚠️  NEUTRAL = {neutral_pct:.1%} — extremamente desbalanceado! "
            f"Considere aumentar os thresholds (sell/buy) no master_config.yaml. "
            f"O Especialista terá poucos sinais de volatilidade para aprender."
        )
    else:
        logger.info(f"📊 Class Distribution: Neutral={neutral_pct:.1%}, Directional={1-neutral_pct:.1%}")

    # Sniper Inheritance: Herdar alphas genéticos do best_params se disponíveis
    class_weights = best_params.get('base_alpha_list') 
    if class_weights is None:
        # Fallback para master_config ou valor padrão
        class_weights = config['training']['specialization_weights'].get('class_weights', [4.85, 0.38, 4.61])
        logger.info(f"⚖️ Alphas Fallback: {class_weights}")
    else:
        logger.info(f"🧬 Alphas Genetic Inheritance: {class_weights}")
    batch_size    = best_params.get('batch_size', 512)
    seq_len       = best_params['seq_len']

    # ── 4. K-Fold Loop ────────────────────────────────────────────────────────
    # -- 4. K-Fold Loop --------------------------------------------------------
    fold_results = []   # list of (test_original_idx, probs, targets)

    for fold_k, (train_idx, test_idx) in enumerate(
        blocked_purged_kfold_indices(n_total, n_splits, purge_bars)
    ):
        logger.info("="*60)
        logger.info(f"[KFOLD] FOLD {fold_k+1}/{n_splits} | Train: {len(train_idx):,} rows | Test: {len(test_idx):,} rows")

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
                    logger.warning(f"[Fold {fold_k}] Left Gap VIOLATION: {gap_left} < {purge_bars} bars!")
                else:
                    logger.info(f"[Fold {fold_k}] Left Purge Gap OK: {gap_left} bars ({gap_left * resample_min} min)")
                    
            if len(right_train) > 0:
                gap_right = right_train.min() - test_max - 1
                if gap_right < purge_bars:
                    logger.warning(f"[Fold {fold_k}] Right Gap VIOLATION: {gap_right} < {purge_bars} bars!")
                else:
                    logger.info(f"[Fold {fold_k}] Right Purge Gap OK: {gap_right} bars ({gap_right * resample_min} min)")

        # ── [v12.4] Sub-Dataset Creation (Memory Safe) ────────────────────────
        import copy
        train_ds_fold = copy.copy(val_dataset_master)
        train_ds_fold.global_indices = val_dataset_master.global_indices[train_idx]
        
        test_ds_fold = copy.copy(val_dataset_master)
        test_ds_fold.global_indices = val_dataset_master.global_indices[test_idx]

        # -- Per-Fold Scaler (Anti-Leakage) ------------------------------------
        scaler = StandardScaler()
        
        # Fitamos apenas com os dados de treino do fold atual
        # Se estiver em modo Lightning (RAM), escalamos tudo agora para performance
        if val_dataset_master.pre_loaded_X is not None:
            X_master_raw = val_dataset_master.pre_loaded_X.numpy()
            N_m, L_m, F_m = X_master_raw.shape
            
            # Fit na fatia de treino (achatada p/ 2D)
            X_train_slice = X_master_raw[train_idx].reshape(-1, F_m)
            scaler.fit(X_train_slice)
            
            # Aplicamos a transformação em todo o bloco mestre p/ este fold
            X_master_scaled = scaler.transform(X_master_raw.reshape(-1, F_m)).reshape(N_m, L_m, F_m)
            
            # Cada dataset do fold recebe o bloco mestre já escalado.
            # O mapeamento via global_indices filtrará as fatias corretas no __getitem__.
            scaled_X_tensor = torch.from_numpy(X_master_scaled)
            train_ds_fold.pre_loaded_X = scaled_X_tensor
            train_ds_fold.pre_loaded_y = val_dataset_master.pre_loaded_y
            
            test_ds_fold.pre_loaded_X = scaled_X_tensor
            test_ds_fold.pre_loaded_y = val_dataset_master.pre_loaded_y
        else:
            # Modo Lazy: Passamos o scaler para ser aplicado no __getitem__
            # Aqui precisaríamos de um fit parcial ou fit em amostra, mas preservando a lógica lazy.
            # No modo atual da cloud, pre_loaded_X sempre existirá p/ 5k amostras.
            train_ds_fold.scaler = scaler
            test_ds_fold.scaler = scaler

        # Save fold scaler
        scaler_path = oof_dir / f"scaler_fold_{fold_k}.pkl"
        with open(scaler_path, 'wb') as sf:
            pickle.dump(scaler, sf)

        # -- Train clone -------------------------------------------------------
        model = train_specialist_fold(
            train_dataset=train_ds_fold,
            val_dataset=test_ds_fold,
            config=config,
            best_params=best_params,
            class_weights=class_weights,
            DEVICE=DEVICE,
            fold_k=fold_k,
        )

        # -- Save Specialist Model Weights -------------------------------------
        model_path = oof_dir / f"model_fold_{fold_k}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"[Fold {fold_k}] Model saved: {model_path.name}")

        # -- OOF Inference (raw logits on unseen test block) -------------------
        probs, targets, valid_meta = run_inference(model, test_ds_fold, DEVICE)
        pred_classes = np.argmax(probs, axis=1)

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

        # Para compatibilidade com salvamento legados: valid_idx simulado
        # Em modo streaming, o original_row_idx é calculado via metadados das sequências.
        fold_df = pd.DataFrame({
            "original_row_idx": valid_meta,
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

    # -- 5. Assemble full_oof.parquet ------------------------------------------
    logger.info("Assembling full_oof.parquet (chronological order)...")

    # Concatenate and sort by original_row_idx to restore chronological order
    all_fold_dfs = [pl.read_parquet(oof_dir / f"fold_{k}.parquet") for k in range(n_splits)]
    full_oof_df  = pl.concat(all_fold_dfs).sort("original_row_idx")

    full_oof_path = oof_dir / "full_oof.parquet"
    full_oof_df.write_parquet(full_oof_path)

    # Final stats
    n_oof = len(full_oof_df)
    coverage = n_oof / n_total
    logger.info(f"full_oof.parquet: {n_oof:,} rows | Coverage: {coverage:.1%} of Foundation Val")
    logger.info(f"Expected: ~{n_total - n_splits * purge_bars:,} (after purge removal)")

    # Validate no duplicate original_row_idx
    n_unique = full_oof_df['original_row_idx'].n_unique()
    if n_unique < n_oof:
        logger.warning(f"{n_oof - n_unique} duplicate rows in full_oof.parquet!")
    else:
        logger.info("No duplicate rows in full_oof.parquet")

    # -- 6. Integrated Security QA ---------------------------------------------
    logger.info("Running Automated Specialist Security QA (pytest)...")
    # Dynamic parallelism for Security QA
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, ImportError, NotImplementedError):
        cpu_count = os.cpu_count() or 1
    pytest_workers = max(1, cpu_count - 1)

    qa_log_path = oof_dir / "kfold_security_QA.log"
    try:
        with open(qa_log_path, 'w', encoding='utf-8') as qa_file:
            subprocess.run(
                ["pytest", "tests/kfold/test_kfold_security.py", "-v", "-n", str(pytest_workers)],
                stdout=qa_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False
            )
        logger.info("[OK] Specialist QA Report saved to " + str(qa_log_path))
    except Exception as e:
        logger.error(f"[FAIL] Specialist QA failed: {e}")
    logger.info("[END] K-Fold Specialist finished. Auditor fusion ready.")


if __name__ == "__main__":
    run_kfold_specialist()
    # Audit Logs -> RESULTADOS_.../AUDITORIA/KFOLD_SPECIALIST/
    _cfg = load_config()
    upload_audit_to_drive(
        local_dirs=["logs/kfold_specialist"],
        stage_name="KFOLD_SPECIALIST",
        config=_cfg,
    )
