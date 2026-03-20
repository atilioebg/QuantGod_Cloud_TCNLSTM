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
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing, compute_alpha_from_labels
from sklearn.preprocessing import StandardScaler
from src.cloud.base_model.utils.logging_utils import setup_logger, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import get_specialized_dir
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

class QuantGodLazyDataset(Dataset):
    """
    SOTA Big Data Loader: Mapeia trilhões de amostras em disco sem carregar na RAM.
    Suporta: Subsampling de Classes, Epoch-Chunking e Island Protection.
    """
    def __init__(self, parquet_files: list, feature_cols: list, seq_len: int, config: dict, is_train: bool = True):
        self.parquet_files = sorted(parquet_files)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.config = config
        self.is_train = is_train

        opt_cfg = config.get('training_optimization', {})
        
        # 1. Mapeamento de Arquivos e Índices
        self.file_offsets = []
        self.global_indices = []
        self.total_raw_rows = 0
        
        logger.info(f"🔍 Scan do Dataset ({'Treino' if is_train else 'Val'}): {len(parquet_files)} arquivos...")
        
        for f_idx, pf in enumerate(self.parquet_files):
            # Scan rápido apenas dos comprimentos (metadata)
            meta = pl.scan_parquet(pf).select(['target', 'island_id']).collect()
            count = len(meta)
            
            targets = meta['target'].to_numpy()
            islands = meta['island_id'].to_numpy()
            
            # Filtro de Island Protection (Lookback não cruza GAP)
            valid_mask = (islands[:count - seq_len + 1] == islands[seq_len - 1:])
            valid_local_indices = np.where(valid_mask)[0]
            
            # 2. Subsampling de Classes (Prado Event-Filtering)
            if is_train and opt_cfg.get('use_class_subsampling', False):
                target_cls = opt_cfg.get('subsample_class_target', 1)
                keep_ratio = opt_cfg.get('subsample_keep_ratio', 0.2)
                
                # Pegamos o target na posição final da sequência
                seq_targets = targets[valid_local_indices + seq_len - 1]
                
                # Sorteio aleatório para a classe alvo
                random_vals = np.random.rand(len(valid_local_indices))
                drop_mask = (seq_targets == target_cls) & (random_vals > keep_ratio)
                valid_local_indices = valid_local_indices[~drop_mask]

            # Registrar mapeamento: global_idx -> (file_idx, local_idx)
            for l_idx in valid_local_indices:
                self.global_indices.append((f_idx, l_idx))
            
            self.total_raw_rows += count

        # 3. Epoch-Chunking (Sorteio de sub-lote representativo)
        if is_train and opt_cfg.get('use_epoch_chunking', False):
            n_samples = opt_cfg.get('samples_per_epoch', 5000000)
            if n_samples < len(self.global_indices):
                selected_indices = np.random.choice(len(self.global_indices), n_samples, replace=False)
                self.global_indices = [self.global_indices[i] for i in selected_indices]
                logger.info(f"✂️ Epoch-Chunking Ativo: Reduzindo {len(self.global_indices)} -> {n_samples} amostras/época.")

        logger.info(f"✅ Dataset carregado: {len(self.global_indices)} sequências válidas.")

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        f_idx, l_idx = self.global_indices[idx]
        pf = self.parquet_files[f_idx]
        
        # Leitura sob demanda (Lazy) usando slice do Polars (rápido com mmap)
        # Nota: Lemos o bloco da sequência [l_idx : l_idx + seq_len]
        df_seq = pl.read_parquet(pf, columns=self.feature_cols + ['target']).slice(l_idx, self.seq_len)
        
        x_seq = df_seq.select(self.feature_cols).to_numpy().astype(np.float32)
        y_label = df_seq.select('target').to_numpy()[-1, 0] # Último da sequência
        
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
    def __init__(self, config, spec_space, train_files, val_files, class_weights, DEVICE, best_params=None):
        self.config = config
        self.spec_space = spec_space
        self.train_files = train_files
        self.val_files = val_files
        self.class_weights = class_weights
        self.DEVICE = DEVICE
        self.best_params = best_params or {}
        
        self.feature_cols = config['model']['feature_names']
        self.epochs = config['optimization']['search_space']['epochs']
        self.patience = config['optimization']['search_space']['early_stopping_patience']
        
        # Optimization Config
        self.opt_cfg = config.get('training_optimization', {})

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

        # 2. Datasets & Loaders (Lazy SOTA)
        train_dataset = QuantGodLazyDataset(self.train_files, self.feature_cols, seq_len, self.config, is_train=True)
        val_dataset = QuantGodLazyDataset(self.val_files, self.feature_cols, seq_len, self.config, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

        # 3. Model
        model = Hybrid_TCN_LSTM(
            num_features=len(self.feature_cols),
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
        spec_cfg = self.config['training'].get('specialization_weights', {})
        search_space = self.config['optimization'].get('search_space', {})

        # -- Optimizer --
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # ── [AMP & GRAD ACCUMULATION] ──────────────────────────────────────────
        use_amp = self.opt_cfg.get('use_amp', True)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        use_grad_acc = self.opt_cfg.get('use_gradient_accumulation', False)
        acc_steps = self.opt_cfg.get('accumulation_steps', 1) if use_grad_acc else 1

        # -- Focal Loss Parameters (Optuna or Fixed) --
        # Gamma
        if spec_cfg.get('spec_optimize_gamma', False):
            gamma = trial.suggest_float("spec_loss_gamma", search_space['spec_loss_gamma'][0], search_space['spec_loss_gamma'][1])
        else:
            gamma = self.best_params.get('base_loss_gamma', spec_cfg.get('spec_gamma', 2.0))
        
        # Smoothing
        if spec_cfg.get('spec_optimize_smoothing', False):
            smoothing = trial.suggest_float("spec_loss_smoothing", search_space['spec_loss_smoothing'][0], search_space['spec_loss_smoothing'][1])
        else:
            smoothing = self.best_params.get('base_loss_smoothing', spec_cfg.get('spec_smoothing', 0.1))

        # Alpha (Class Weights)
        if spec_cfg.get('spec_optimize_class_weights', False):
            a_side = trial.suggest_float("spec_alpha_side", search_space['spec_alpha_side'][0], search_space['spec_alpha_side'][1])
            a_neu  = trial.suggest_float("spec_alpha_neutral", search_space['spec_alpha_neutral'][0], search_space['spec_alpha_neutral'][1])
            alpha  = torch.tensor([a_side, a_neu, a_side], dtype=torch.float32).to(self.DEVICE)
        elif spec_cfg.get('spec_use_auto_class_weights', True):
            # For LazyDataset, we calculate Alpha once or use pre-set. Standard logic for Specialist is AUTO.
            alpha = torch.tensor([4.85, 0.38, 4.61], dtype=torch.float32).to(self.DEVICE)
        else:
            class_weights = spec_cfg.get('spec_class_weights', [4.85, 0.38, 4.61])
            alpha = torch.tensor(class_weights, dtype=torch.float32).to(self.DEVICE)

        # ── Trial Start Log ────────────
        logger.info(f"Trial {trial.number} START | batch={batch_size}, seq={seq_len}, "
                    f"lr={lr:.6f}, alpha=[{alpha[0]:.2f}, {alpha[1]:.2f}, {alpha[2]:.2f}] | AccSteps={acc_steps}")

        best_sniper_score = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            # Training Phase
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(self.DEVICE), yb.to(self.DEVICE)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(xb)
                    loss = FocalLossWithSmoothing(outputs, yb, alpha=alpha, gamma=gamma, smoothing=smoothing)
                    loss = loss / acc_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * acc_steps
            
            # Validation Phase
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                for xb, yb in val_loader:
                    xb = xb.to(self.DEVICE)
                    outputs = model(xb)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(yb.numpy())
            
        if spec_cfg.get('spec_optimize_gamma', False):
            gamma = trial.suggest_float("spec_loss_gamma", search_space['spec_loss_gamma'][0], search_space['spec_loss_gamma'][1])
        else:
            gamma = self.best_params.get('base_loss_gamma')
            if gamma is None:
                gamma = spec_cfg.get('spec_gamma', 2.0)
        
        # -- Label Smoothing --
        if spec_cfg.get('spec_optimize_smoothing', False):
            smoothing = trial.suggest_float("spec_loss_smoothing", search_space['spec_loss_smoothing'][0], search_space['spec_loss_smoothing'][1])
        else:
            smoothing = self.best_params.get('base_loss_smoothing')
            if smoothing is None:
                smoothing = spec_cfg.get('spec_smoothing', 0.1)

        # ── Trial Start Log (To include Focal parameters) ────────────
        a_str = f"[{alpha[0]:.2f}, {alpha[1]:.2f}, {alpha[2]:.2f}]"
        logger.info(f"Trial {trial.number} START | tcn={tcn_channels}, lstm={lstm_hidden}, "
                    f"layers={num_lstm_layers}, batch={batch_size}, seq={seq_len}, "
                    f"drop={dropout:.4f}, lr={lr:.6f}, wd={weight_decay:.4f} | "
                    f"alpha={a_str}, gamma={gamma:.2f}, smooth={smoothing:.2f}")

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
    spec_cfg = config['training'].get('specialization_weights', {})
    use_dir = spec_cfg.get('use_best_f1_dir', False)
    param_file = "best_dir_params.json" if use_dir else "best_params.json"
    
    foundation_params_path = Path("src/cloud/base_model/otimizacao") / param_file
    if not foundation_params_path.exists():
        logger.error(f"❌ {param_file} missing at {foundation_params_path}. Run base optimization first.")
        sys.exit(1)
        
    with open(foundation_params_path, 'r', encoding='utf-8') as f:
        best_foundation_params = json.load(f)
        
    # 2. Build Specialist Space
    spec_space = build_specialist_space(config, best_foundation_params)
    
    # 3. Scan Files (Lazy Path)
    spec_train_dir = Path(get_specialized_dir(config)) / "train"
    spec_val_dir   = Path(get_specialized_dir(config)) / "val"
    
    train_files = sorted(list(spec_train_dir.glob("*.parquet")))
    val_files   = sorted(list(spec_val_dir.glob("*.parquet")))
    
    if not train_files:
        logger.error(f"❌ No training parquets in {spec_train_dir}")
        sys.exit(1)
        
    class_weights = config['training']['specialization_weights'].get('class_weights', [4.85, 0.38, 4.61])
    n_trials = config['optimization'].get('n_trials_specialist', 40)
    
    logger.info(f"🎯 Starting Optimized Specialist Training (Lazy Loading Mode). Trials: {n_trials}")

    objective = SpecialistObjective(
        config=config, 
        spec_space=spec_space,
        train_files=train_files, val_files=val_files,
        class_weights=class_weights,
        DEVICE=DEVICE,
        best_params=best_foundation_params
    )
    
    study_name = "quantgod_specialist_v2_optimized"
    storage_name = "sqlite:///optuna_specialist.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction="maximize"
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"✅ Optimization finished. Best Sniper Score: {study.best_value:.4f}")
    logger.info(f"🏆 Best Hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
        
    # Prepare Final Dictionary and Inject Global Focal Logs if not optimized
    final_params = study.best_params.copy()
    spec_cfg = config['training'].get('specialization_weights', {})
    
    # 1. Class Weights (Alpha) fallback injection
    if not list(filter(lambda k: "alpha" in k, final_params.keys())):
        if spec_cfg.get('spec_use_auto_class_weights', True):
            # Recalculate OOF weights just for logging context if Auto is selected
            final_alpha = compute_alpha_from_labels(y_train_raw, num_classes=3, device=torch.device("cpu")).tolist()
        else:
            final_alpha = spec_cfg.get('spec_class_weights', [4.85, 0.38, 4.61])
        final_params['spec_alpha_list'] = final_alpha

    # 2. Gamma / Smoothing fallback injection (Smart Inheritance)
    if "spec_loss_gamma" not in final_params:
        gamma = spec_cfg.get('spec_gamma')
        if gamma is None: gamma = best_foundation_params.get('base_loss_gamma', 2.0)
        final_params['spec_loss_gamma'] = gamma

    if "spec_loss_smoothing" not in final_params:
        smooth = spec_cfg.get('spec_smoothing')
        if smooth is None: smooth = best_foundation_params.get('base_loss_smoothing', 0.1)
        final_params['spec_loss_smoothing'] = smooth

    # Save best parameters
    best_params_path = Path("src/cloud/base_model/otimizacao/best_params_specialist.json")
    with open(best_params_path, 'w') as f:
        json.dump(final_params, f, indent=4)
        
    # Save the physical model of the best trial
    best_trial = study.best_trial
    if "best_model_state" in best_trial.user_attrs:
        model_output_path = Path(project_root) / config['pipeline_paths']['best_specialized_model']
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_trial.user_attrs["best_model_state"], model_output_path)
        logger.info(f"💾 Best Specialized Model Saved to: {model_output_path}")

if __name__ == "__main__":
    run_specialization()
