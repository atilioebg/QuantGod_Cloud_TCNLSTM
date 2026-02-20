
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

log_dir = Path("logs/optimization")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _setup_log_file(labelled_dir: str):
    """Adds FileHandler with dynamic name: optimization_{suffix}_{timestamp}.log"""
    dir_name = Path(labelled_dir).name
    suffix = dir_name.replace("labelled_", "") if dir_name.startswith("labelled_") else dir_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"optimization_{suffix}_{timestamp}.log"
    fh = logging.FileHandler(log_filename, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log file: {log_filename}")


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X; self.y = y; self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx:idx + self.seq_len]),
                torch.tensor(self.y[idx + self.seq_len - 1], dtype=torch.long))


def load_data(labelled_dir, feature_cols):
    parquet_files = sorted(list(Path(labelled_dir).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No labelled data in {labelled_dir}")
    dfs = [pl.read_parquet(pf, columns=feature_cols + ['target']) for pf in parquet_files]
    return pl.concat(dfs), feature_cols


def objective(trial, X_all, y_all, config, class_weights):
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

        # ── Data split (chronological 80/20) ──────────────────────────────────
        split_idx = int(len(X_all) * 0.8)
        train_dataset = SequenceDataset(X_all[:split_idx], y_all[:split_idx], seq_len)
        val_dataset   = SequenceDataset(X_all[split_idx:], y_all[split_idx:], seq_len)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)
        val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)

        # ── Model ──────────────────────────────────────────────────────────────
        model = Hybrid_TCN_LSTM(
            num_features=X_all.shape[1],
            seq_len=seq_len,
            tcn_channels=tcn_channels,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers,
            num_classes=3,
            dropout=dropout,
        ).to(DEVICE)

        # ── Loss from centralized class_weights (single source of truth) ───────
        weights   = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        amp_scaler = torch.amp.GradScaler('cuda')

        # ── Training loop ──────────────────────────────────────────────────────
        best_val_f1 = 0.0
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
            current_lr  = scheduler.get_last_lr()[0]

            logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_loss/len(train_loader):.4f} | "
                        f"Val Loss: {val_loss/len(val_loader):.4f} | "
                        f"F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f} | "
                        f"F1 [SELL/NEU/BUY]: [{f1_per_cls[0]:.3f}/{f1_per_cls[1]:.3f}/{f1_per_cls[2]:.3f}] | "
                        f"LR: {current_lr:.6f}")

            if f1_macro > best_val_f1:
                best_val_f1 = f1_macro

            trial.report(f1_macro, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                del model, train_loader, val_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        # Cleanup after trial
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        return best_val_f1

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
    # ── Load configs ──────────────────────────────────────────────────────────
    optuna_cfg_path = Path("src/cloud/base_model/otimizacao/optimization_config.yaml")
    with open(optuna_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    base_cfg_path = Path("src/cloud/base_model/configs/base_model_config.yaml")
    with open(base_cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # class_weights from single source of truth
    class_weights = base_cfg['training']['class_weights']
    feature_cols  = base_cfg['model']['feature_names']

    _setup_log_file(config['paths']['labelled_dir'])

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data for optimization...")
    df, _ = load_data(config['paths']['labelled_dir'], feature_cols)
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_all = df.select('target').to_numpy().flatten().astype(np.int64)
    logger.info(f"Data loaded: {X_raw.shape} | class_weights: {class_weights}")

    # Normalize fit on train only
    split_idx = int(len(X_raw) * 0.8)
    scaler = StandardScaler()
    scaler.fit(X_raw[:split_idx])
    X_all = scaler.transform(X_raw).astype(np.float32)

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
    study.optimize(
        lambda trial: objective(trial, X_all, y_all, config, class_weights),
        n_trials=config['optimization']['n_trials'],
        timeout=config['optimization']['timeout'],
    )

    logger.info(f"Optimization complete | Best F1 Macro: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    out_path = Path("src/cloud/base_model/otimizacao/best_params.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best params saved: {out_path}")


if __name__ == "__main__":
    run_optimization()
