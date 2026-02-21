
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

project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing, compute_alpha_from_labels

log_dir = Path("logs/optimization")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


from src.cloud.base_model.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


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


def objective(trial, X_train, y_train, X_val, y_val, config, class_weights):
    """
    Optuna objective function for TCN+LSTM hyperparameter search.

    Engineering constraints enforced:
    - class_weights loaded from centralized base_model_config.yaml (not hardcoded)
    - OOM intercepted: torch.cuda.empty_cache() + TrialPruned (graceful skip)
    - Gradient clipping (norm=1.0) applied on every backward pass
    """
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Search space ───────────────────────────────────────────────────────
        tcn_channels    = trial.suggest_categorical("tcn_channels",    config['search_space']['tcn_channels'])
        lstm_hidden     = trial.suggest_categorical("lstm_hidden",      config['search_space']['lstm_hidden'])
        num_lstm_layers = trial.suggest_int("num_lstm_layers",
                                            min(config['search_space']['num_lstm_layers']),
                                            max(config['search_space']['num_lstm_layers']))
        batch_size      = trial.suggest_categorical("batch_size",       config['search_space']['batch_size'])
        dropout         = trial.suggest_float("dropout",
                                              config['search_space']['dropout'][0],
                                              config['search_space']['dropout'][1])
        seq_len         = trial.suggest_categorical("seq_len",          config['search_space']['seq_len'])
        lr              = trial.suggest_float("lr",
                                              config['search_space']['lr'][0],
                                              config['search_space']['lr'][1], log=True)
        epochs          = config['search_space']['epochs']

        logger.info(f"Trial {trial.number} START | tcn={tcn_channels}, lstm={lstm_hidden}, "
                    f"layers={num_lstm_layers}, batch={batch_size}, seq={seq_len}, "
                    f"drop={dropout:.3f}, lr={lr:.6f}")

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

        # ── Loss: dynamic alpha per trial (inverse-frequency from train labels) ─
        alpha     = compute_alpha_from_labels(y_train, num_classes=3, device=DEVICE)
        criterion = FocalLossWithSmoothing(alpha=alpha, gamma=2.0, smoothing=0.1)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        amp_scaler = torch.amp.GradScaler('cuda')

        # ── Training loop ──────────────────────────────────────────────────────
        best_macro_f1 = 0.0   # Champion tracker: Best F1 Macro
        best_dir_f1   = 0.0   # Champion tracker: Best F1 Direcional (SELL+BUY)
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for b_idx, (batch_X, batch_y) in enumerate(train_loader):
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
                if (b_idx + 1) % 100 == 0:
                    pct = (b_idx + 1) / len(train_loader) * 100
                    logger.info(f"Trial {trial.number} | Epoch {epoch+1} | "
                                f"Batch {b_idx+1}/{len(train_loader)} ({pct:.1f}%) | "
                                f"Loss: {loss.item():.4f}")
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

            f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1_macro    = f1_score(all_targets, all_preds, average='macro',    zero_division=0)
            f1_per_cls  = f1_score(all_targets, all_preds, average=None,       zero_division=0)
            f1_dir      = (f1_per_cls[0] + f1_per_cls[2]) / 2
            current_lr  = scheduler.get_last_lr()[0]

            logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_loss/len(train_loader):.4f} | "
                        f"Val Loss: {val_loss/len(val_loader):.4f} | "
                        f"F1 Macro: {f1_macro:.4f} | F1 Dir: {f1_dir:.4f} | "
                        f"F1 [SELL/NEU/BUY]: [{f1_per_cls[0]:.3f}/{f1_per_cls[1]:.3f}/{f1_per_cls[2]:.3f}] | "
                        f"LR: {current_lr:.6f}")

            # ── Dual Champion Tracking (GLOBAL — across all trials) ───────────
            # MACRO global best: query Optuna study's own best value (f1_macro objective)
            try:
                global_best_macro = trial.study.best_value   # best f1_macro seen so far
            except ValueError:
                global_best_macro = 0.0  # no completed trial yet

            if f1_macro > global_best_macro:
                macro_save_path = Path("data/models/best_tcn_lstm.pt")
                macro_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), macro_save_path)
                logger.info(f"🥇 [MACRO]  Trial {trial.number} | Global F1 Macro record: {f1_macro:.4f} "
                            f"(prev: {global_best_macro:.4f}) → saved best_tcn_lstm.pt")

            # DIR global best: check max best_f1_dir stored across completed trials
            completed = [t for t in trial.study.trials
                         if t.state.name == "COMPLETE" and "best_f1_dir" in t.user_attrs]
            global_best_dir = max((t.user_attrs["best_f1_dir"] for t in completed), default=0.0)

            if f1_dir > global_best_dir:
                dir_save_path = Path("data/models/best_tcn_lstm_dir.pt")
                dir_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), dir_save_path)
                logger.info(f"🏆 [DIR]    Trial {trial.number} | Global F1 Dir record: {f1_dir:.4f} "
                            f"(prev: {global_best_dir:.4f}) → saved best_tcn_lstm_dir.pt")

            # Update this trial's running best_f1_dir attribute for ranking later
            if f1_dir > trial.user_attrs.get("best_f1_dir", 0.0):
                trial.set_user_attr("best_f1_dir", f1_dir)

            # Optimization target: F1 Macro
            trial.report(f1_macro, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                del model, train_loader, val_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        # Cleanup after trial
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        return best_macro_f1  # Optuna ranks trials by this value (F1 Macro)

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
    if len(sys.argv) > 1:
        optuna_cfg_path = Path(sys.argv[1])
    else:
        optuna_cfg_path = Path("src/cloud/base_model/otimizacao/optimization_config.yaml")

    with open(optuna_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    base_cfg_path = Path("src/cloud/base_model/configs/base_model_config.yaml")
    with open(base_cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # class_weights from single source of truth
    class_weights = base_cfg['training']['class_weights']
    feature_cols  = base_cfg['model']['feature_names']

    # ── Suffix Extraction & Logging Setup ──────────────────────────────────
    suffix = ""
    train_dir_path = Path(config['paths']['train_dir'])
    import re
    match = re.search(r"(_SELL_.*)$", str(train_dir_path.parent))
    if match:
        suffix = match.group(1)
        
    setup_logger("optimization", suffix)

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

    # ── Optuna study ──────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=config['paths']['study_name'],
        storage=config['paths']['db_path'],
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    logger.info(f"Starting {config['optimization']['n_trials']} trials | "
                f"Metric: {config['optimization']['metric']} | "
                f"Timeout: {config['optimization']['timeout']}s")
    
    start_trials = len(study.trials)
    start_time = datetime.now()

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, config, class_weights),
        n_trials=config['optimization']['n_trials'],
        timeout=config['optimization']['timeout'],
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    trials_run = len(study.trials) - start_trials

    # Inference of stopping reason
    if duration >= (config['optimization']['timeout'] - 60): # 1 minute tolerance
        stop_reason = f"TIMEOUT ALCANÇADO (As {config['optimization']['timeout']}s expiraram)"
    elif trials_run >= config['optimization']['n_trials']:
        stop_reason = f"MÁXIMO DE TRIALS ({config['optimization']['n_trials']}) ALCANÇADOS"
    else:
        stop_reason = "PARADA MANUAL OU ERRO INTERNO"

    logger.info("="*60)
    logger.info(f"🛑 OTIMIZAÇÃO FINALIZADA 🛑")
    logger.info(f"Motivo da Parada: {stop_reason}")
    logger.info(f"Tempo de Execução da Sessão: {duration/3600:.2f} Horas")
    logger.info(f"Trials Executados nesta Sessão: {trials_run}")
    logger.info("="*60)

    logger.info(f"Optimization complete | Melhor F1 Macro: {study.best_trial.value:.4f}")
    logger.info(f"Melhores Parametros Macro: {study.best_params}")

    # ── Save MACRO champion params (trial ranked by study objective) ─────────
    out_params_path = Path("src/cloud/base_model/otimizacao/best_params.json")
    with open(out_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"🥇 [MACRO] Best params saved: {out_params_path}")

    # ── Save DIRECTIONAL champion params (trial with highest best_f1_dir attr) 
    completed = [t for t in study.trials if t.state.name == "COMPLETE"
                 and "best_f1_dir" in t.user_attrs]
    if completed:
        best_dir_trial = max(completed, key=lambda t: t.user_attrs["best_f1_dir"])
        best_dir_params = best_dir_trial.params
        best_dir_val    = best_dir_trial.user_attrs["best_f1_dir"]
        logger.info(f"🏆 [DIR]   Best trial: {best_dir_trial.number} | F1 Dir: {best_dir_val:.4f}")
        logger.info(f"🏆 [DIR]   Best params: {best_dir_params}")
        out_dir_path = Path("src/cloud/base_model/otimizacao/best_dir_params.json")
        with open(out_dir_path, "w") as f:
            json.dump(best_dir_params, f, indent=4)
        logger.info(f"🏆 [DIR]   Best params saved: {out_dir_path}")
    else:
        logger.warning("⚠️ No completed trials with f1_dir attribute found. best_dir_params.json not updated.")


if __name__ == "__main__":
    run_optimization()
