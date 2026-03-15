
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import polars as pl
import numpy as np
import yaml
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import subprocess
import os

project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing, compute_alpha_from_labels

# Tracking variables for cross-trial real-time logging
GLOBAL_BEST_MACRO = -1.0
GLOBAL_BEST_DIR   = -1.0

# ── Logging Setup ──────────────────────────────────────────────────────────
from src.cloud.base_model.utils.logging_utils import setup_logger, setup_optuna_logging
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths

# Stop Optuna's default logger from duplicating messages natively
setup_optuna_logging()

# Initial dummy logger (will be properly set up in run_optimization)
logger = logging.getLogger("optimization")

class SequenceDataset(torch.utils.data.Dataset):
    """Island-aware sliding window dataset. Mirrors run_specialization.py exactly.
    Sequences that cross island boundaries are excluded to prevent cross-gap leakage."""
    def __init__(self, X, y, island_ids, seq_len):
        self.X, self.y, self.seq_len = X, y, seq_len
        max_idx = len(X) - seq_len
        if max_idx < 0:
            self.valid_indices = np.array([], dtype=np.int64)
        else:
            self.valid_indices = np.where(
                island_ids[:max_idx + 1] == island_ids[seq_len - 1:]
            )[0].astype(np.int64)
        logger.info(f"SequenceDataset: {len(self.valid_indices)}/{max(0, max_idx+1)} valid sequences (Lookback protection active).")

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        return (torch.from_numpy(self.X[i:i + self.seq_len]),
                torch.tensor(self.y[i + self.seq_len - 1], dtype=torch.long))


def load_data(directory, feature_cols):
    parquet_files = sorted(list(Path(directory).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No labelled data in {directory}")
    dfs = []
    for i, pf in enumerate(parquet_files):
        df_i = pl.read_parquet(pf, columns=feature_cols + ['target', 'island_id'])
        df_i = df_i.with_columns(pl.col('island_id') + (i * 10000))
        dfs.append(df_i)
    df = pl.concat(dfs)
    logger.info(f"Loaded {len(df):,} rows from {directory}")
    return df, feature_cols


def objective(trial, X_train, y_train, island_train, X_val, y_val, island_val, config, base_cfg, auto_alphas=None):
    """
    Optuna objective function for TCN+LSTM hyperparameter search.

    Engineering constraints enforced:
    - class_weights loaded from centralized master_config.yaml (not hardcoded)
    - OOM intercepted: torch.cuda.empty_cache() + TrialPruned (graceful skip)
    - Gradient clipping (norm=1.0) applied on every backward pass
    """
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Search space ───────────────────────────────────────────────────────
        tcn_channels    = trial.suggest_categorical("tcn_channels",    config['optimization']['search_space']['tcn_channels'])
        lstm_hidden     = trial.suggest_categorical("lstm_hidden",      config['optimization']['search_space']['lstm_hidden'])
        num_lstm_layers = trial.suggest_int("num_lstm_layers",
                                            min(config['optimization']['search_space']['num_lstm_layers']),
                                            max(config['optimization']['search_space']['num_lstm_layers']))
        batch_size      = trial.suggest_categorical("batch_size",       config['optimization']['search_space']['batch_size'])
        dropout         = trial.suggest_float("dropout",
                                              config['optimization']['search_space']['dropout'][0],
                                              config['optimization']['search_space']['dropout'][1])
        seq_len         = trial.suggest_categorical("seq_len",          config['optimization']['search_space']['seq_len'])
        lr              = trial.suggest_float("lr",
                                              config['optimization']['search_space']['lr'][0],
                                              config['optimization']['search_space']['lr'][1], log=True)
        weight_decay    = trial.suggest_float("weight_decay",
                                              config['optimization']['search_space']['weight_decay'][0],
                                              config['optimization']['search_space']['weight_decay'][1], log=True)
        epochs          = config['optimization']['search_space']['epochs']

        # Move Start Logger below Loss computation to include their values

        # ── Datasets ───────────────────────────────────────────────────────────
        train_dataset = SequenceDataset(X_train, y_train, island_train, seq_len)
        val_dataset   = SequenceDataset(X_val, y_val, island_val, seq_len)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)
        val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)

        # ── Model ──────────────────────────────────────────────────────────────
        model = Hybrid_TCN_LSTM(
            num_features=X_train.shape[1],
            seq_len=seq_len,
            tcn_channels=tcn_channels,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers,
            num_classes=3,
            dropout=dropout,
        ).to(DEVICE)

        # ── Loss: dynamic alpha per trial or manual weights from config ─────────
        foundation_cfg = config['training'].get('foundation_weights', {})
        search_space = config['optimization']['search_space']
        
        # 1. Class Weights (Alpha)
        if foundation_cfg.get('base_use_dynamic_range_optuna', False) and auto_alphas is not None:
            dyn_cfg = config['optimization'].get('dynamic_range_config', {})
            mult = dyn_cfg.get('range_multiplier', 0.5)
            floor = dyn_cfg.get('min_floor_alpha', 0.05)
            
            # center = auto_alphas[c]
            # low = max(floor, center * (1 - mult))
            # high = center * (1 + mult)
            a_sell = trial.suggest_float("base_alpha_sell", max(floor, auto_alphas[0].item() * (1 - mult)), auto_alphas[0].item() * (1 + mult))
            a_neu  = trial.suggest_float("base_alpha_neutral", max(floor, auto_alphas[1].item() * (1 - mult)), auto_alphas[1].item() * (1 + mult))
            a_buy  = trial.suggest_float("base_alpha_buy", max(floor, auto_alphas[2].item() * (1 - mult)), auto_alphas[2].item() * (1 + mult))
            alpha = torch.tensor([a_sell, a_neu, a_buy], dtype=torch.float32).to(DEVICE)
        elif foundation_cfg.get('base_optimize_class_weights', False):
            a_side = trial.suggest_float("base_alpha_side", search_space['base_alpha_side'][0], search_space['base_alpha_side'][1])
            a_neu  = trial.suggest_float("base_alpha_neutral", search_space['base_alpha_neutral'][0], search_space['base_alpha_neutral'][1])
            alpha  = torch.tensor([a_side, a_neu, a_side], dtype=torch.float32).to(DEVICE)
        elif foundation_cfg.get('base_use_auto_class_weights', True):
            alpha = compute_alpha_from_labels(y_train, num_classes=3, device=DEVICE)
        else:
            class_weights = foundation_cfg.get('base_class_weights', [1.0, 1.0, 1.0])
            alpha = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
            
        # 2. Gamma
        if foundation_cfg.get('base_optimize_gamma', False):
            gamma = trial.suggest_float("base_loss_gamma", search_space['base_loss_gamma'][0], search_space['base_loss_gamma'][1])
        else:
            gamma = foundation_cfg.get('base_gamma', 2.0)
            
        # 3. Label Smoothing
        if foundation_cfg.get('base_optimize_smoothing', False):
            smoothing = trial.suggest_float("base_loss_smoothing", search_space['base_loss_smoothing'][0], search_space['base_loss_smoothing'][1])
        else:
            smoothing = foundation_cfg.get('base_smoothing', 0.1)

        # 4. Sniper Loss (Directional Penalty)
        use_sniper = foundation_cfg.get('base_use_sniper_loss', False)
        sniper_weight = foundation_cfg.get('base_sniper_weight', 1.0)

        # ── Trial Start Log (Moved here to include Focal parameters) ────────────
        a_str = f"[{alpha[0]:.2f}, {alpha[1]:.2f}, {alpha[2]:.2f}]"
        s_str = f"ON (w={sniper_weight})" if use_sniper else "OFF"
        logger.info(f"Trial {trial.number} START | tcn={tcn_channels}, lstm={lstm_hidden}, "
                    f"layers={num_lstm_layers}, batch={batch_size}, seq={seq_len}, "
                    f"drop={dropout:.4f}, lr={lr:.6f}, wd={weight_decay:.4f} | "
                    f"alpha={a_str}, gamma={gamma:.2f}, smooth={smoothing:.2f}, sniper={s_str}")

        criterion = FocalLossWithSmoothing(
            alpha=alpha, 
            gamma=gamma, 
            smoothing=smoothing,
            use_sniper=use_sniper,
            sniper_weight=sniper_weight
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        amp_scaler = torch.amp.GradScaler('cuda')

        # ── Training loop ──────────────────────────────────────────────────────
        best_macro_f1 = 0.0   # Champion tracker: Best F1 Macro
        best_dir_f1   = 0.0   # Champion tracker: Best F1 Direcional (SELL+BUY)
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = config['optimization']['search_space'].get('early_stopping_patience', 6)
        base_metric_name = config['optimization'].get('base_metric', 'f1_macro')

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            from tqdm import tqdm
            # In non-TTY environments (RunPod logs), reduce tqdm volume
            is_tty = sys.stdout.isatty()
            pbar = tqdm(
                train_loader, 
                desc=f"Trial {trial.number} | Epoch {epoch+1}", 
                leave=False,
                mininterval=10.0 if not is_tty else 0.1,  # 10s updates in logs
                maxinterval=100.0 if not is_tty else 10.0
            )
            for b_idx, (batch_X, batch_y) in enumerate(pbar):
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
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
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            pbar.close()
            scheduler.step()

            # Validation
            model.eval()
            all_preds, all_targets = [], []
            val_loss = 0.0
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                        out = model(batch_X)
                        val_loss += criterion(out["logits"], batch_y).item()
                        preds = torch.argmax(out["logits"], dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(batch_y.cpu().numpy())

            # Calculation of metrics
            f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            f1_per_cls = f1_score(all_targets, all_preds, average=None, zero_division=0)
            if len(f1_per_cls) < 3:
                # Padding case
                tmp = np.zeros(3)
                for i, val in enumerate(f1_per_cls): tmp[i] = val
                f1_per_cls = tmp
            f1_dir = np.mean([f1_per_cls[0], f1_per_cls[2]])
            # Penalidade por classe zerada (Zero Penalty)
            # Sniper Guard: Se Sell (0) ou Buy (2) for 0, o modelo é severamente punido.
            has_zero_signal = (f1_per_cls[0] == 0 or f1_per_cls[2] == 0)
            has_zero_class  = np.any(f1_per_cls == 0)
            
            # Se tiver signal zerado, a punição é ABSOLUTA (0.0). Para o Optuna, esse modelo é nulo.
            penalty = 0.0 if has_zero_signal else (0.5 if has_zero_class else 1.0)

            # Aplica penalidade em ambas as métricas
            f1_macro = f1_macro * penalty
            f1_dir   = f1_dir * penalty

            # ── Dead Model Guard: Economia de GPU ──────────────────────────
            # Se após 10 épocas o modelo não produz sinal em ambos os lados, aborta.
            if epoch >= 9 and has_zero_signal:
                logger.info(f"🚫 Trial {trial.number} ABORTADO: Modelo 'Cérebro Morto' (Signal Zero na Época {epoch+1})")
                del model, train_loader, val_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                return 0.0  # Retorna nota zero para o Optuna descartar a região


            # ── Sniper Metric Adaptation: Minimum F1 optimization ──────────
            use_min_f1 = config['training']['foundation_weights'].get('base_use_min_f1_optimization', False)
            if use_min_f1:
                # O objetivo principal passa a ser o elo mais fraco, também punido
                orig_f1_macro = f1_macro
                f1_macro = float(np.min(f1_per_cls)) * penalty
                
            current_lr = scheduler.get_last_lr()[0]
            current_val_loss = val_loss / len(val_loader)


            # ── Epoch Summary ─────────────────────────────────────────────────────────
            p_status = "⚠️ ZERO_PENALTY" if has_zero_class else "✅ OK"
            m_label = "F1 M MIN" if use_min_f1 else "F1 M"
            # ASCII cleanup for status
            p_status = "[WARN] ZERO_PENALTY" if has_zero_class else "[OK]"
            logger.info(
                f"T{trial.number} E{epoch+1}/{epochs} | {p_status} | "
                f"L: {train_loss/len(train_loader):.8f}/{current_val_loss:.8f} | "
                f"{m_label}: {f1_macro:.8f} D: {f1_dir:.8f} | "
                f"[S/N/B]: [{f1_per_cls[0]:.4f}/{f1_per_cls[1]:.4f}/{f1_per_cls[2]:.4f}] | "
                f"LR: {current_lr:.8f}"
            )

            # ── Local Champion Tracking (Within this Trial) ───────────────────
            # v4.9: Patience now tracks the OPTIMIZATION METRIC (F1), not Loss.
            # This resolves the "Patience Incongruence" and gives the model more time to stabilize.
            # Metric value depends on master_config base_metric selection.
            trial_improved = False
            if base_metric_name == 'f1_macro':
                if f1_macro > best_macro_f1:
                    trial_improved = True
            else:
                if f1_dir > best_dir_f1:
                    trial_improved = True
            
            if trial_improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if f1_macro > best_macro_f1:
                best_macro_f1 = f1_macro
            if f1_dir > best_dir_f1:
                best_dir_f1 = f1_dir

            # ── Dual Champion Tracking (GLOBAL — across all trials and epochs) ────
            global GLOBAL_BEST_MACRO, GLOBAL_BEST_DIR

            # MACRO global best
            if f1_macro > GLOBAL_BEST_MACRO and f1_macro > 0:
                prev_macro = GLOBAL_BEST_MACRO
                GLOBAL_BEST_MACRO = f1_macro
                from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_drive
                base_dir = get_drive_session_path("MODELOS", config)
                model_path = resolve_local_drive(Path(base_dir) / config['pipeline_paths']['best_tcn_lstm_model'])
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(model_path))
                m_rec_label = "F1 M MIN" if use_min_f1 else "F1 Macro"
                logger.info(f"[BEST MACRO]  Trial {trial.number} | Global {m_rec_label} record: {f1_macro:.8f} "
                            f"(prev: {prev_macro:.8f}) -> saved {model_path.name}")

            # DIR global best
            if f1_dir > GLOBAL_BEST_DIR and f1_dir > 0:
                prev_dir = GLOBAL_BEST_DIR
                GLOBAL_BEST_DIR = f1_dir
                from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_drive
                base_dir = get_drive_session_path("MODELOS", config)
                dir_save_path = resolve_local_drive(Path(base_dir) / config['pipeline_paths']['best_tcn_lstm_dir_model'])
                dir_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(dir_save_path))
                logger.info(f"[BEST DIR]    Trial {trial.number} | Global F1 Dir record: {f1_dir:.8f} "
                            f"(prev: {prev_dir:.8f}) -> saved {dir_save_path.name}")

            # Update this trial's running best_f1_dir attribute for ranking later
            if f1_dir >= trial.user_attrs.get("best_f1_dir", -1.0):
                trial.set_user_attr("best_f1_dir", f1_dir)

            # ── Optimization Target Update ──────────────────────────────────
            trial_metric_val = f1_macro if base_metric_name == 'f1_macro' else f1_dir
            trial_best_val = best_macro_f1 if base_metric_name == 'f1_macro' else best_dir_f1
            
            # Optimization target
            trial.report(trial_metric_val, epoch)
            
            # ── Sniper Alpha Early Stopping ──────────────────────────────────
            if patience_counter >= patience_limit:
                logger.info(f"Trial {trial.number} stopped early due to patience limit ({patience_limit} epochs without improvement)")
                del model, train_loader, val_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                return trial_best_val

            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned by Optuna at epoch {epoch+1}")
                del model, train_loader, val_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        # Cleanup after trial
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        trial_best_val = best_macro_f1 if config['optimization'].get('base_metric', 'f1_macro') == 'f1_macro' else best_dir_f1
        return trial_best_val  # Optuna ranks trials by this value

    except RuntimeError as e:
        # ── CRITICAL: OOM guard (Constraint #4) ───────────────────────────────
        # Catches CUDA out of memory for any combination that exceeds VRAM.
        # Immediately clears residual VRAM and gracefully skips the trial.
        if "out of memory" in str(e).lower():
            logger.warning(f"Trial {trial.number} — CUDA OOM! Clearing cache and pruning...")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        raise e


def run_optimization():
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    
    with open(master_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # config logic from master source of truth
    foundation_cfg = config['training'].get('foundation_weights', {})
    class_weights = foundation_cfg.get('class_weights', [1.0, 1.0, 1.0])
    feature_cols  = config['model']['feature_names']
    
    # ensure "paths" dict exists just in case it's nested or we use config directly
    if 'paths' not in config:
        config['paths'] = {}
        config['paths']['train_dir'] = 'AUTO'
        config['paths']['val_dir'] = 'AUTO'

    # ── Suffix Extraction & Logging Setup ──────────────────────────────────
    suffix = ""
    train_dir_path = Path(config['paths']['train_dir'])
    import re
    match = re.search(r"(_SELL_.*)$", str(train_dir_path.parent))
    if match:
        suffix = match.group(1)
        
    setup_logger("optimization", suffix)

    # ── Resolve AUTO paths ──────────────────────────────────────────────────
    config['paths']['train_dir'], config['paths']['val_dir'] = resolve_data_paths(config['paths'])
    logger.info(f"📁 DYNAMIC DATA PATHS: Train={config['paths']['train_dir']} | Val={config['paths']['val_dir']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data for optimization...")
    train_df, _ = load_data(config['paths']['train_dir'], feature_cols)
    val_df, _   = load_data(config['paths']['val_dir'],   feature_cols)

    X_train_raw   = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train       = train_df.select('target').to_numpy().flatten().astype(np.int64)
    island_train  = train_df.select('island_id').to_numpy().flatten()
    X_val_raw     = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val         = val_df.select('target').to_numpy().flatten().astype(np.int64)
    island_val    = val_df.select('island_id').to_numpy().flatten()

    # Normalize fit on train only
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    
    from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_drive
    base_dir = get_drive_session_path("MODELOS", config)
    scaler_path = resolve_local_drive(Path(base_dir) / config['pipeline_paths']['scaler_foundation'])
    
    import joblib
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"💾 Scaler saved properly to: {scaler_path}")

    # ── Log Alpha Class Weights Globally ─────────────────────────────────────
    import torch
    dummy_device = torch.device("cpu")
    auto_alphas = compute_alpha_from_labels(y_train, num_classes=3, device=dummy_device)
    
    if foundation_cfg.get('base_use_dynamic_range_optuna', False):
        logger.info(f"FocalLoss alpha: DYNAMIC RANGE active anchor points: [S:{auto_alphas[0]:.2f}, N:{auto_alphas[1]:.2f}, B:{auto_alphas[2]:.2f}]")
    elif foundation_cfg.get('base_optimize_class_weights', False):
        logger.info("FocalLoss alpha: Optuna assumira o controle Dinamico no espaco de busca (Static).")
    elif foundation_cfg.get('base_use_auto_class_weights', True):
        logger.info(f"FocalLoss alpha (AUTO computed from foundation labels): {auto_alphas.tolist()}")
    else:
        logger.info(f"FocalLoss alpha (MANUAL from config): {class_weights}")
        
    gamma = foundation_cfg.get('base_gamma', 2.0)
    smoothing = foundation_cfg.get('base_smoothing', 0.1)
    
    gamma_str = "OPTUNA_DINAMICO" if foundation_cfg.get('base_optimize_gamma', False) else f"{gamma}"
    smooth_str = "OPTUNA_DINAMICO" if foundation_cfg.get('base_optimize_smoothing', False) else f"{smoothing}"
    sniper_status = "ENABLED" if foundation_cfg.get('base_use_sniper_loss', False) else "DISABLED"
    logger.info(f"Loss: FocalLossWithSmoothing | fallback_gamma={gamma_str} | fallback_smoothing={smooth_str} | SniperLoss={sniper_status}")


    # ── Optuna study ──────────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(n_startup_trials=30, multivariate=True)
    study = optuna.create_study(
        study_name=config['optimization']['study_name'],
        storage=config['pipeline_paths']['db_path'],
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    # Initialize global trackers from study history (if resuming)
    global GLOBAL_BEST_MACRO, GLOBAL_BEST_DIR
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    
    # Path verification: if the physical files don't exist, we must ignore DB records 
    # to ensure the current session saves its own best models.
    from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_drive
    base_dir = get_drive_session_path("MODELOS", config)
    macro_path = resolve_local_drive(Path(base_dir) / config['pipeline_paths']['best_tcn_lstm_model'])
    dir_path   = resolve_local_drive(Path(base_dir) / config['pipeline_paths']['best_tcn_lstm_dir_model'])
    
    if completed and macro_path.exists():
        GLOBAL_BEST_MACRO = study.best_value
        logger.info(f"Resuming study. Macro Record found: {GLOBAL_BEST_MACRO:.4f}")
    else:
        GLOBAL_BEST_MACRO = 0.0
        if completed: logger.info("DB has records but physical .pt is missing. Starting fresh Macro save-track.")

    if completed and dir_path.exists():
        GLOBAL_BEST_DIR = max((t.user_attrs.get("best_f1_dir", 0.0) for t in completed), default=0.0)
        logger.info(f"Resuming study. Dir Record found: {GLOBAL_BEST_DIR:.4f}")
    else:
        GLOBAL_BEST_DIR = 0.0
        if completed: logger.info("DB has records but physical _dir.pt is missing. Starting fresh Dir save-track.")

    metric_to_max = config['optimization']['base_metric']

    if foundation_cfg.get('base_use_min_f1_optimization', False) and metric_to_max == 'f1_macro':
        metric_to_max = "MIN(Sell, Neu, Buy)"
    
    logger.info(f"Starting {config['optimization']['n_trials']} trials | "
                f"Metric: {metric_to_max} | "
                f"Timeout: {config['optimization']['timeout']}s")
    
    start_trials = len(study.trials)
    start_time = datetime.now()

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, island_train, X_val, y_val, island_val, config, config, auto_alphas=auto_alphas),
        n_trials=config['optimization']['n_trials'],
        timeout=config['optimization']['timeout'],
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    trials_run = len(study.trials) - start_trials

    # Inference of stopping reason
    if duration >= (config['optimization']['timeout'] - 60): # 1 minute tolerance
        stop_reason = f"TIMEOUT ALCANCADO (As {config['optimization']['timeout']}s expiraram)"
    elif trials_run >= config['optimization']['n_trials']:
        stop_reason = f"MAXIMO DE TRIALS ({config['optimization']['n_trials']}) ALCANCADOS"
    else:
        stop_reason = "PARADA MANUAL OU ERRO INTERNO"

    logger.info("="*60)
    logger.info(f"--- OTIMIZACAO FINALIZADA ---")
    logger.info(f"Motivo da Parada: {stop_reason}")
    logger.info(f"Tempo de Execucao da Sessao: {duration/3600:.2f} Horas")
    logger.info(f"Trials Executados nesta Sessao: {trials_run}")
    # ── Champion Param Extraction Logic (Unified) ───────────────────────────
    def extract_full_params(trial_obj, foundation_cfg_node, labels_ref, n_feats):
        params = trial_obj.params.copy()
        params['num_features'] = n_feats  # Save architecture input size
        
        # 1. Alphas (Genetic Inheritance)
        if any(k in params for k in ["base_alpha_sell", "base_alpha_buy"]):
            params['base_alpha_list'] = [
                float(params.get('base_alpha_sell', 1.0)),
                float(params.get('base_alpha_neutral', 1.0)),
                float(params.get('base_alpha_buy', 1.0))
            ]
        elif "base_alpha_side" in params:
            params['base_alpha_list'] = [
                float(params['base_alpha_side']),
                float(params['base_alpha_neutral']),
                float(params['base_alpha_side'])
            ]
        else:
            # Fallback to auto-computed or manual config
            if foundation_cfg_node.get('base_use_auto_class_weights', True):
                params['base_alpha_list'] = compute_alpha_from_labels(labels_ref, num_classes=3, device=torch.device("cpu")).tolist()
            else:
                params['base_alpha_list'] = foundation_cfg_node.get('base_class_weights', [1.0, 1.0, 1.0])
        
        # 2. Gamma & Smoothing
        if "base_loss_gamma" not in params:
            params['base_loss_gamma'] = foundation_cfg_node.get('base_gamma', 2.0)
        if "base_loss_smoothing" not in params:
            params['base_loss_smoothing'] = foundation_cfg_node.get('base_smoothing', 0.1)
            
        return params

    logger.info("="*60)

    use_min_f1 = config['training']['foundation_weights'].get('base_use_min_f1_optimization', False)
    m_f_label = "F1 M MIN" if use_min_f1 else "F1 Macro"
    
    logger.info(f"[MACRO] Best trial: {study.best_trial.number} | {m_f_label}: {study.best_trial.value:.8f}")
    
    final_params = extract_full_params(study.best_trial, foundation_cfg, y_train, X_train.shape[1])
    
    formatted_best_params = {k: f"{v:.8f}" if isinstance(v, float) else v for k, v in final_params.items()}
    logger.info(f"[MACRO] Best params: {formatted_best_params}")

    # ── Save MACRO champion params
    out_params_path = Path("src/cloud/base_model/otimizacao") / "best_params.json"
    with open(out_params_path, "w", encoding='utf-8') as f:
        json.dump(final_params, f, indent=4, ensure_ascii=False)
    logger.info(f"[BEST MACRO] Best params saved: {out_params_path}")

    # ── Downstream Pipeline Automation ─────────────────────────────────────
    # We no longer auto-update master_config.yaml to avoid Git conflicts.
    # Downstream scripts (specialization, feature importance) now load
    # parameters dynamically from best_params.json.

    # ── Save DIRECTIONAL champion params (trial with highest best_f1_dir attr) 
    completed = [t for t in study.trials if t.state.name == "COMPLETE"
                 and "best_f1_dir" in t.user_attrs]
    if completed:
        best_dir_trial = max(completed, key=lambda t: t.user_attrs["best_f1_dir"])
        best_dir_params = extract_full_params(best_dir_trial, foundation_cfg, y_train, X_train.shape[1])

        best_dir_val = best_dir_trial.user_attrs["best_f1_dir"]
        logger.info(f"[BEST DIR]   Best trial: {best_dir_trial.number} | F1 Dir: {best_dir_val:.8f}")
        
        out_dir_path = Path("src/cloud/base_model/otimizacao") / "best_dir_params.json"
        with open(out_dir_path, "w", encoding='utf-8') as f:
            json.dump(best_dir_params, f, indent=4, ensure_ascii=False)
        logger.info(f"[BEST DIR]   Best params saved: {out_dir_path}")
    else:
        logger.warning("[WARN] No completed trials with f1_dir attribute found. best_dir_params.json not updated.")

    # ── Pipeline Execution Finished ──────────────────────────────────────────
    logger.info("="*60)
    logger.info("[OK] Optuna Foundation Training completed. Orchestration is now handled by run_manager.py")
    logger.info("="*60)

if __name__ == "__main__":
    run_optimization()
