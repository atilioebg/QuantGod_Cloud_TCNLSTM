
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
GLOBAL_BEST_MACRO = 0.0
GLOBAL_BEST_DIR   = 0.0

# ── Logging Setup ──────────────────────────────────────────────────────────
from src.cloud.base_model.utils.logging_utils import setup_logger, setup_optuna_logging
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths

# Stop Optuna's default logger from duplicating messages natively
setup_optuna_logging()

# Initial dummy logger (will be properly set up in run_optimization)
logger = logging.getLogger("optimization")

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X; self.y = y; self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx:idx + self.seq_len]),
                torch.tensor(self.y[idx + self.seq_len - 1], dtype=torch.long))


def load_data(directory, feature_cols):
    parquet_files = sorted(list(Path(directory).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No labelled data in {directory}")
    dfs = [pl.read_parquet(pf, columns=feature_cols + ['target']) for pf in parquet_files]
    df = pl.concat(dfs)
    logger.info(f"Loaded {len(df):,} rows from {directory}")
    return df, feature_cols


def objective(trial, X_train, y_train, X_val, y_val, config, base_cfg):
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

        logger.info(f"Trial {trial.number} START | tcn={tcn_channels}, lstm={lstm_hidden}, "
                    f"layers={num_lstm_layers}, batch={batch_size}, seq={seq_len}, "
                    f"drop={dropout:.8f}, lr={lr:.8f}, wd={weight_decay:.8f}")

        # ── Datasets ───────────────────────────────────────────────────────────
        train_dataset = SequenceDataset(X_train, y_train, seq_len)
        val_dataset   = SequenceDataset(X_val, y_val, seq_len)
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
        if foundation_cfg.get('use_auto_class_weights', True):
            alpha = compute_alpha_from_labels(y_train, num_classes=3, device=DEVICE)
        else:
            class_weights = foundation_cfg.get('class_weights', [1.0, 1.0, 1.0])
            alpha = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        
        gamma = foundation_cfg.get('gamma', 2.0)
        smoothing = foundation_cfg.get('smoothing', 0.1)
        criterion = FocalLossWithSmoothing(alpha=alpha, gamma=gamma, smoothing=smoothing)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        amp_scaler = torch.amp.GradScaler('cuda')

        # ── Training loop ──────────────────────────────────────────────────────
        best_macro_f1 = 0.0   # Champion tracker: Best F1 Macro
        best_dir_f1   = 0.0   # Champion tracker: Best F1 Direcional (SELL+BUY)
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = config['optimization'].get('early_stopping_patience', 3)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}", leave=False)
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

            f1_macro   = f1_score(all_targets, all_preds, average='macro',    zero_division=0)
            f1_per_cls = f1_score(all_targets, all_preds, average=None,       zero_division=0, labels=[0, 1, 2])
            f1_dir     = (f1_per_cls[0] + f1_per_cls[2]) / 2
            current_lr = scheduler.get_last_lr()[0]
            current_val_loss = val_loss / len(val_loader)

            # ── Epoch Summary ─────────────────────────────────────────────────────────
            logger.info(
                f"T{trial.number} E{epoch+1}/{epochs} | "
                f"L: {train_loss/len(train_loader):.8f}/{current_val_loss:.8f} | "
                f"F1 M: {f1_macro:.8f} D: {f1_dir:.8f} | "
                f"[S/N/B]: [{f1_per_cls[0]:.4f}/{f1_per_cls[1]:.4f}/{f1_per_cls[2]:.4f}] | "
                f"LR: {current_lr:.8f}"
            )

            # ── Local Champion Tracking (Within this Trial) ───────────────────
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
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
            if f1_macro > GLOBAL_BEST_MACRO:
                prev_macro = GLOBAL_BEST_MACRO
                GLOBAL_BEST_MACRO = f1_macro
                macro_save_path = Path("data/models/best_tcn_lstm.pt")
                macro_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), macro_save_path)
                logger.info(f"🥇 [MACRO]  Trial {trial.number} | Global F1 Macro record: {f1_macro:.8f} "
                            f"(prev: {prev_macro:.8f}) → saved best_tcn_lstm.pt")

            # DIR global best
            if f1_dir > GLOBAL_BEST_DIR:
                prev_dir = GLOBAL_BEST_DIR
                GLOBAL_BEST_DIR = f1_dir
                dir_save_path = Path("data/models/best_tcn_lstm_dir.pt")
                dir_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), dir_save_path)
                logger.info(f"🏆 [DIR]    Trial {trial.number} | Global F1 Dir record: {f1_dir:.8f} "
                            f"(prev: {prev_dir:.8f}) → saved best_tcn_lstm_dir.pt")

            # Update this trial's running best_f1_dir attribute for ranking later
            if f1_dir > trial.user_attrs.get("best_f1_dir", 0.0):
                trial.set_user_attr("best_f1_dir", f1_dir)

            base_metric_name = config['optimization'].get('base_metric', 'f1_macro')
            trial_metric_val = f1_macro if base_metric_name == 'f1_macro' else f1_dir
            trial_best_val = best_macro_f1 if base_metric_name == 'f1_macro' else best_dir_f1
            
            # Optimization target
            trial.report(trial_metric_val, epoch)
            
            # ── Sniper Alpha Early Stopping ──────────────────────────────────
            if patience_counter >= patience_limit:
                logger.info(f"Trial {trial.number} stopped early due to patience ({patience_limit} epochs without improvement)")
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

    X_train_raw = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train     = train_df.select('target').to_numpy().flatten().astype(np.int64)
    X_val_raw   = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val       = val_df.select('target').to_numpy().flatten().astype(np.int64)

    # Normalize fit on train only
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    
    import joblib
    scaler_path = Path(config['pipeline_paths']['scaler_foundation'])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"💾 Scaler saved properly to: {scaler_path}")

    # ── Log Alpha Class Weights Globally ─────────────────────────────────────
    import torch
    dummy_device = torch.device("cpu")
    if foundation_cfg.get('use_auto_class_weights', True):
        # Fail fast approach if auto fails here.
        alpha_base = compute_alpha_from_labels(y_train, num_classes=3, device=dummy_device)
        logger.info(f"FocalLoss alpha (AUTO computed from foundation labels): {alpha_base.tolist()}")
    else:
        logger.info(f"FocalLoss alpha (MANUAL from config): {class_weights}")
        
    gamma = foundation_cfg.get('gamma', 2.0)
    smoothing = foundation_cfg.get('smoothing', 0.1)
    logger.info(f"Loss: FocalLossWithSmoothing | gamma={gamma} | smoothing={smoothing}")


    # ── Optuna study ──────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=config['optimization']['study_name'],
        storage=config['pipeline_paths']['db_path'],
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    # Initialize global trackers from study history (if resuming)
    global GLOBAL_BEST_MACRO, GLOBAL_BEST_DIR
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if completed:
        GLOBAL_BEST_MACRO = study.best_value
        GLOBAL_BEST_DIR = max((t.user_attrs.get("best_f1_dir", 0.0) for t in completed), default=0.0)
        logger.info(f"Resuming study. Current records: Macro={GLOBAL_BEST_MACRO:.4f}, Dir={GLOBAL_BEST_DIR:.4f}")

    logger.info(f"Starting {config['optimization']['n_trials']} trials | "
                f"Metric: {config['optimization']['base_metric']} | "
                f"Timeout: {config['optimization']['timeout']}s")
    
    start_trials = len(study.trials)
    start_time = datetime.now()

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, config, config),
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
    logger.info("="*60)

    logger.info(f"Optimization complete | Melhor F1 Macro: {study.best_trial.value:.8f}")
    
    # Clean precision formatted string for logger
    formatted_best_params = {k: f"{v:.8f}" if isinstance(v, float) else v for k, v in study.best_params.items()}
    logger.info(f"Melhores Parametros Macro: {formatted_best_params}")

    # ── Save MACRO champion params (trial ranked by study objective) ─────────
    out_params_path = Path("src/cloud/base_model/otimizacao") / "best_params.json"
    with open(out_params_path, "w", encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=4, ensure_ascii=False)
    logger.info(f"🥇 [MACRO] Best params saved: {out_params_path}")

    # ── Auto-update master_config.yaml ─────────────────────────────────────
    if master_cfg_path.exists():
        try:
            with open(master_cfg_path, 'r', encoding='utf-8') as f:
                train_cfg_dict = yaml.safe_load(f)
            
            if 'training' not in train_cfg_dict:
                train_cfg_dict['training'] = {}
            if 'hyperparameters' not in train_cfg_dict['training']:
                train_cfg_dict['training']['hyperparameters'] = {}
                
            for k, v in study.best_params.items():
                train_cfg_dict['training']['hyperparameters'][k] = v
                
            with open(master_cfg_path, 'w', encoding='utf-8') as f:
                yaml.dump(train_cfg_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"🔄 Updated {master_cfg_path} with MACRO best params.")
        except Exception as e:
            logger.error(f"❌ Failed to auto-update master_config.yaml: {e}")

    # ── Save DIRECTIONAL champion params (trial with highest best_f1_dir attr) 
    completed = [t for t in study.trials if t.state.name == "COMPLETE"
                 and "best_f1_dir" in t.user_attrs]
    if completed:
        best_dir_trial = max(completed, key=lambda t: t.user_attrs["best_f1_dir"])
        best_dir_params = best_dir_trial.params
        best_dir_val    = best_dir_trial.user_attrs["best_f1_dir"]
        formatted_dir_params = {k: f"{v:.8f}" if isinstance(v, float) else v for k, v in best_dir_params.items()}
        logger.info(f"🏆 [DIR]   Best trial: {best_dir_trial.number} | F1 Dir: {best_dir_val:.8f}")
        logger.info(f"🏆 [DIR]   Best params: {formatted_dir_params}")
        out_dir_path = Path("src/cloud/base_model/otimizacao") / "best_dir_params.json"
        with open(out_dir_path, "w", encoding='utf-8') as f:
            json.dump(best_dir_params, f, indent=4, ensure_ascii=False)
        logger.info(f"🏆 [DIR]   Best params saved: {out_dir_path}")
    else:
        logger.warning("⚠️ No completed trials with f1_dir attribute found. best_dir_params.json not updated.")

    # ── Pipeline Automation ───────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("--- INICIANDO TRANSFERENCIA BASE (FOUNDATION) ---")
    
    # 0. Descobrir o nome exato do arquivo de log (.log) gerado pelo setup_logger
    log_filename = "optimization.log" # Fallback
    opt_logger = logging.getLogger("optimization")
    
    # Busca o FileHandler atrelado ao logger
    for handler in opt_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_filename = Path(handler.baseFilename).name
            break
            
    logger.info(f"Log-ancora identificado para a transferencia: {log_filename}")
    
    # 0.5 Feature Importance (Optuna Best Model)
    if config['optimization'].get('run_feature_importance_after', False):
        logger.info("-> Analisando Feature Importance do Melhor Modelo do Optuna...")
        try:
            # Roda o feature importance focando no modelo campeao MACRO da base
            env = os.environ.copy()
            env["QUIET_LOGGING"] = "1"
            subprocess.run([sys.executable, "src/cloud/base_model/treino/feature_importance.py", "data/models/best_tcn_lstm.pt"], 
                           check=True, env=env)
        except subprocess.CalledProcessError as e:
            logger.error(f"⚠️ Feature importance falhou (o pipeline continuara): {e}")

    # 1. Transfer Foundation
    try:
        env = os.environ.copy()
        env["QUIET_LOGGING"] = "1"
        subprocess.run([sys.executable, "src/cloud/base_model/utils/transfer.py", log_filename, "foundation"], 
                       check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Falha no transfer base (foundation): {e}")
        
    # 2. Check if Specialization should run
    if config['optimization'].get('run_specialized_after', False):
        logger.info("="*60)
        logger.info("--- INICIANDO PIPELINE DE ESPECIALIZACAO AUTOMATICA ---")
        
        try:
            # 2.1 Create Splits
            dataset_path = str(Path(config['paths']['train_dir']).parent)
            logger.info("-> 1/3 Gerando Splits Especializados...")
            env = os.environ.copy()
            env["QUIET_LOGGING"] = "1"
            subprocess.run([sys.executable, "src/cloud/base_model/treino/create_specialized_splits.py", dataset_path], 
                           check=True, env=env)
            
            # 2.2 Run Specialization
            logger.info("-> 2/3 Treinando Modelo Especialista...")
            subprocess.run([sys.executable, "src/cloud/base_model/treino/run_specialization.py"], 
                           check=True, env=env)
            
            # 2.3 Transfer Specialized
            logger.info("-> 3/3 Transferindo Especialista pro Drive...")
            subprocess.run([sys.executable, "src/cloud/base_model/utils/transfer.py", log_filename, "specialized"], 
                           check=True, env=env)
            
            logger.info("PIPELINE 100%% COMPLETO E EXECUTADO COM SUCESSO!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Falha na cascata de especializacao: {e}")
    else:
        logger.info("--- Pipeline parado no Foundation (run_specialized_after = False/Nao configurado) ---")
    logger.info("="*60)


if __name__ == "__main__":
    run_optimization()
