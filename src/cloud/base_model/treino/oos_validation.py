import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import polars as pl
import numpy as np
import yaml
import logging
import pickle
import argparse
from pathlib import Path
import sys
import json
from sklearn.metrics import f1_score, classification_report

# Project root on sys.path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """Island-aware sliding window dataset. Mirrors run_specialization.py exactly."""
    def __init__(self, X: np.ndarray, y: np.ndarray, island_ids: np.ndarray, seq_len: int):
        self.X, self.y, self.seq_len = X, y, seq_len
        max_idx = len(X) - seq_len
        if max_idx < 0:
            self.valid_indices = np.array([], dtype=np.int64)
        else:
            self.valid_indices = np.where(
                island_ids[:max_idx + 1] == island_ids[seq_len - 1:]
            )[0].astype(np.int64)
        logger.info(f"SequenceDataset: {len(self.valid_indices)}/{max(0, max_idx+1)} valid sequences (Lookback protection active).")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x_seq   = self.X[i: i + self.seq_len]
        y_label = self.y[i + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.long)

def load_data(directory: str, feature_cols: list):
    parquet_files = sorted(list(Path(directory).glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    dfs = []
    for i, pf in enumerate(parquet_files):
        df_i = pl.read_parquet(pf, columns=feature_cols + ['target', 'island_id'])
        df_i = df_i.with_columns(pl.col('island_id') + (i * 10000))
        dfs.append(df_i)
    df = pl.concat(dfs)
    logger.info(f"Loaded {len(df):,} rows from {directory}")
    return df

def run_oos_validation(test_dir: str, permutation: bool = False):
    setup_logger("oos_validation", "")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Configurations
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        master_cfg = yaml.safe_load(f)
    
    feature_cols = master_cfg['model']['feature_names']
    
    # Try to load best params from Optuna if available, else fallback
    params_filename = "best_params.json"
    best_params_path = Path("src/cloud/base_model/otimizacao") / params_filename
    
    tcn_channels = 128
    lstm_hidden = 128
    num_lstm_layers = 2
    dropout = 0.5
    seq_len = 360
    batch_size = 1024

    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        tcn_channels = best_params.get('tcn_channels', tcn_channels)
        lstm_hidden = best_params.get('lstm_hidden', lstm_hidden)
        num_lstm_layers = best_params.get('num_lstm_layers', num_lstm_layers)
        dropout = best_params.get('dropout', dropout)
        seq_len = best_params.get('seq_len', seq_len)
        batch_size = best_params.get('batch_size', batch_size)
        logger.info(f"✨ Loaded hyperparameters from {params_filename}")
    else:
        logger.warning("⚠️ No best_params.json found. Using default architecture parameters.")

    # 2. Load Model and Scaler
    model_path = Path("data/models/treino_best_model.pt") # Specialization champion
    if not model_path.exists():
        model_path = Path("data/models/best_tcn_lstm.pt") # Optuna champion fallback
    
    scaler_path = Path("data/models/treino_scaler_finetuning.pkl")
    
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}.")
        return
    if not scaler_path.exists():
        logger.error(f"❌ Scaler not found at {scaler_path}. Specialization must be run first.")
        return

    # Initialize Model
    model = Hybrid_TCN_LSTM(
        num_features=len(feature_cols),
        seq_len=seq_len,
        tcn_channels=tcn_channels,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_lstm_layers,
        num_classes=3,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    logger.info(f"✅ Loaded model weights from {model_path}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✅ Loaded Scaler")

    # 3. Load Data
    logger.info(f"Loading Out-of-Sample (OOS) Test Data from {test_dir}...")
    try:
        test_df = load_data(test_dir, feature_cols)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return

    X_test_raw = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_test     = test_df.select('target').to_numpy().flatten().astype(np.int64)
    island_test = test_df.select('island_id').to_numpy().flatten()

    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # 4. Standard Evaluation
    def evaluate(X, y, island_ids):
        dataset = SequenceDataset(X, y, island_ids, seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_preds, all_targets = [], []
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch_X, batch_y in loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    out = model(batch_X)
                    preds = torch.argmax(out["logits"], dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
        return all_targets, all_preds

    logger.info("Running standard OOS evaluation...")
    y_true, y_pred = evaluate(X_test, y_test, island_test)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    logger.info(f"🏆 OOS F1 Macro: {f1_macro:.4f}")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=["SELL", "NEUTRAL", "BUY"]))

    # 5. Permutation Test (If enabled)
    if permutation:
        logger.info("\n--- 🕵️ PERMUTATION TEST FOR STRUCTURAL LEAKAGE ---")
        logger.info("Shuffling features randomly to destroy strict chronological patterns...")
        
        # We shuffle the feature columns entirely across time
        # If F1 remains high, the target is leaked statically in the features!
        X_shuffled = X_test.copy()
        
        for col_idx in range(X_shuffled.shape[1]):
            np.random.shuffle(X_shuffled[:, col_idx])
            
        logger.info("Re-evaluating model with shuffled features...")
        y_true_perm, y_pred_perm = evaluate(X_shuffled, y_test, island_test)
        
        f1_macro_perm = f1_score(y_true_perm, y_pred_perm, average='macro', zero_division=0)
        logger.info(f"⚠️ PERMUTED F1 Macro: {f1_macro_perm:.4f}")
        
        if f1_macro_perm > 0.38:
            logger.error("🚨 CRITICAL LEAKAGE DETECTED! Model performs exceptionally well even with random noise. A structural lookahead leak is highly probable.")
        else:
            logger.info("✅ SUCCESS: Model performance degraded to random noise/baseline as expected. No static leakage detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOS Validation and Permutation Test")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the unseen test split data directory.")
    parser.add_argument("--permutation", action="store_true", help="Run the permutation feature shuffling test.")
    args = parser.parse_args()
    
    run_oos_validation(args.test_dir, args.permutation)
