
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import polars as pl

# Ensure project root is in path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.utils.logging_utils import setup_logger
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths

logger = setup_logger("feature_importance")

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X; self.y = y; self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx:idx + self.seq_len]),
                torch.tensor(self.y[idx + self.seq_len - 1], dtype=torch.long))

def load_validation_data(val_dir, feature_cols):
    parquet_files = sorted(list(Path(val_dir).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No validation data found in {val_dir}")
    
    # Load all files to get a representative importance
    dfs = [pl.read_parquet(pf, columns=feature_cols + ['target']) for pf in parquet_files]
    df = pl.concat(dfs)
    logger.info(f"Loaded {len(df):,} rows for importance analysis.")
    return df

def calculate_permutation_importance(model, val_loader, feature_cols, device):
    model.eval()
    baseline_f1 = evaluate_model(model, val_loader, device)
    logger.info(f"Baseline Weighted F1: {baseline_f1:.4f}")
    
    importances = {}
    
    # We do a batch-level permutation for speed if the dataset is large,
    # or a full dataset permutation if memory allows. 
    # For TCN+LSTM, we need to shuffle the feature across the entire X array before creating segments.
    
    X_val = val_loader.dataset.X.copy()
    y_val = val_loader.dataset.y
    seq_len = val_loader.dataset.seq_len
    batch_size = val_loader.batch_size
    
    for i, col in enumerate(feature_cols):
        logger.info(f"Evaluating Permutation Importance for: {col}")
        
        # 1. Shuffle only this feature
        X_shuffled = X_val.copy()
        np.random.shuffle(X_shuffled[:, i])
        
        # 2. Create temporary dataset/loader
        shuffled_dataset = SequenceDataset(X_shuffled, y_val, seq_len)
        shuffled_loader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # 3. Measure impact
        shuffled_f1 = evaluate_model(model, shuffled_loader, device)
        importance = baseline_f1 - shuffled_f1
        importances[col] = importance
        logger.info(f" - F1 Drop: {importance:.6f}")
        
    return importances

def evaluate_model(model, loader, device):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device)
            out = model(X)
            preds = torch.argmax(out["probs"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    return f1_score(all_targets, all_preds, average='weighted')

def analyze_tcn_weights(model, feature_cols):
    """
    Extracts importance from the first TCN layer weights.
    Weights shape: (out_channels, num_features, kernel_size)
    """
    first_layer = model.tcn[0].causal_conv.conv
    # Absolute mean weights per input feature
    weights = first_layer.weight.abs().detach().cpu() # (out_ch, in_ch, k)
    importance = weights.mean(dim=(0, 2)).numpy() # Average over out_channels and kernel_size
    
    # Normalize to 0-1
    if importance.max() > 0:
        importance = importance / importance.max()
        
    return dict(zip(feature_cols, importance))

def run_importance_analysis(model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Configs
    with open("src/cloud/base_model/configs/base_model_config.yaml", 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    with open("src/cloud/base_model/otimizacao/optimization_config.yaml", 'r', encoding='utf-8') as f:
        opt_cfg = yaml.safe_load(f)
    with open("src/cloud/base_model/treino/training_config.yaml", 'r', encoding='utf-8') as f:
        train_cfg = yaml.safe_load(f)
        
    feature_cols = base_cfg['model']['feature_names']
    hparams = train_cfg.get('hyperparameters', {})
    
    # 2. Resolve Paths
    _, val_dir = resolve_data_paths(opt_cfg['paths'])
    
    # 3. Load Model
    if model_path is None:
        # Check current config for output macro path
        model_path = Path(train_cfg['paths'].get('model_output_macro', "data/models/best_tcn_lstm.pt"))
        if not model_path.exists():
            logger.error(f"No model checkpoint found at {model_path}")
            return
            
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle state_dict key or direct state_dict
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    # Initialize model with optimized params
    model = Hybrid_TCN_LSTM(
        num_features=len(feature_cols),
        num_classes=base_cfg['model']['num_classes'],
        tcn_channels=hparams.get('tcn_channels', 64), 
        lstm_hidden=hparams.get('lstm_hidden', 256),
        num_lstm_layers=hparams.get('num_lstm_layers', 2),
        seq_len=hparams.get('seq_len', 720),
        dropout=hparams.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(state_dict)
    
    # 4. Load Data
    val_df = load_validation_data(val_dir, feature_cols)
    X_raw = val_df[feature_cols].to_numpy().astype(np.float32)
    y_raw = val_df['target'].to_numpy().astype(np.int64)
    
    # Scaling (Must use same scaler as training)
    scaler_path = Path(train_cfg['paths'].get('scaler_output_macro', "data/models/scaler_finetuning.pkl"))
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        X_val = scaler.transform(X_raw)
    else:
        logger.warning(f"No scaler found at {scaler_path}. Results may be biased.")
        X_val = X_raw

    val_dataset = SequenceDataset(X_val, y_raw, hparams.get('seq_len', 720))
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

    # 5. Permutation Importance
    logger.info("=== CALCULAR PERMUTATION IMPORTANCE ===")
    perm_importance = calculate_permutation_importance(model, val_loader, feature_cols, device)
    
    # 6. TCN Weight Analysis
    logger.info("=== ANALIZAR TCN WEIGHTS (PRIMEIRA CAMADA) ===")
    weight_importance = analyze_tcn_weights(model, feature_cols)
    
    # 7. Summary
    results = pd.DataFrame({
        'Feature': feature_cols,
        'Permutation_Importance': [perm_importance[f] for f in feature_cols],
        'TCN_Weight_Importance': [weight_importance[f] for f in feature_cols]
    }).sort_values(by='Permutation_Importance', ascending=False)
    
    print("\n📊 FEATURE IMPORTANCE REPORT")
    print("-" * 50)
    print(results.to_string(index=False))
    
    output_path = Path("docs/reports/feature_importance.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Report saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_importance_analysis(sys.argv[1])
    else:
        run_importance_analysis()
