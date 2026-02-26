import torch
from torch.utils.data import DataLoader, Dataset
import polars as pl
import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
import sys
import pickle
import sys
import pickle
import json

project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.utils.logging_utils import setup_logger
from src.cloud.base_model.utils.experiment_utils import resolve_data_paths

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    def __len__(self):
        return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_label = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.long)

def load_config():
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
            probs = torch.softmax(outputs["logits"], dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    return np.vstack(all_probs), np.concatenate(all_targets)

def load_and_predict(config, val_dir, context_dir, output_dir):
    # ── Load Context Data ──────────────────────────────────────────────────────
    context_files = sorted(list(Path(context_dir).glob("*.parquet")))
    if not context_files:
        logger.error(f"❌ Context files not found in {context_dir}. Run auditor_preprocessing first.")
        return
        
    df_context_list = []
    for cf in context_files:
        df_c = pl.read_parquet(cf).to_pandas()
        df_context_list.append(df_c)
    full_context_df = pd.concat(df_context_list, ignore_index=True)
    
    # ── Load Raw Feature Data (For Inference) ──────────────────────────────────
    feature_cols = config['model']['feature_names']
    val_files = sorted(list(Path(val_dir).glob("*.parquet")))
    dfs_val = [pl.read_parquet(vf, columns=feature_cols + ['target']) for vf in val_files]
    df_val = pl.concat(dfs_val).to_pandas()
    
    seq_len = config['optimization']['search_space']['seq_len'][0]
    X_val_raw = df_val[feature_cols].to_numpy().astype(np.float32)
    y_val_raw = df_val['target'].to_numpy().astype(np.int64)
    
    # ── Normalization ────────────────────────────────────────────────────────
    scaler_foundation_path = Path(config['pipeline_paths']['scaler_foundation'])
    scaler_specialized_path = Path(config['pipeline_paths']['scaler_specialized'])
    
    with open(scaler_foundation_path, 'rb') as f:
        scaler_base = pickle.load(f)
    with open(scaler_specialized_path, 'rb') as f:
        scaler_spec = pickle.load(f)
        
    X_val_base_norm = scaler_base.transform(X_val_raw).astype(np.float32)
    X_val_spec_norm = scaler_spec.transform(X_val_raw).astype(np.float32)
    
    # ── Datasets ──────────────────────────────────────────────────────────────
    dataset_base = SequenceDataset(X_val_base_norm, y_val_raw, seq_len)
    dataset_spec = SequenceDataset(X_val_spec_norm, y_val_raw, seq_len)
    
    loader_base = DataLoader(dataset_base, batch_size=2048, shuffle=False, num_workers=4)
    loader_spec = DataLoader(dataset_spec, batch_size=2048, shuffle=False, num_workers=4)
    
    # ── Models Loading ────────────────────────────────────────────────────────
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("loading Base and Specialist Models...")
    
    # ── Reconstrução Dinâmica da Arquitetura via best_params.json ─────────────
    # Conforme solicitado, utilizaremos os parâmetros gerados naotimização em vez do índice [0] hardcoded.
    base_params_path = Path("src/cloud/base_model/otimizacao/best_params.json")
    spec_params_path = Path("src/cloud/base_model/otimizacao/best_params_specialist.json")
    
    if not base_params_path.exists() or not spec_params_path.exists():
        logger.error("❌ best_params.json ou best_params_specialist.json não encontrados. Rode a otimização primeiro.")
        return
        
    with open(base_params_path, 'r') as f:
        base_params = json.load(f)
    with open(spec_params_path, 'r') as f:
        spec_params = json.load(f)

    num_features = len(feature_cols)

    model_base = Hybrid_TCN_LSTM(
        num_features=num_features, 
        seq_len=base_params['seq_len'], 
        tcn_channels=base_params['tcn_channels'], 
        lstm_hidden=base_params['lstm_hidden'], 
        num_lstm_layers=base_params['num_lstm_layers'], 
        num_classes=3, 
        dropout=base_params['dropout']
    ).to(DEVICE)
    
    model_spec = Hybrid_TCN_LSTM(
        num_features=num_features, 
        seq_len=spec_params['seq_len'], 
        tcn_channels=spec_params['tcn_channels'], 
        lstm_hidden=spec_params['lstm_hidden'], 
        num_lstm_layers=spec_params['num_lstm_layers'], 
        num_classes=3, 
        dropout=spec_params['dropout']
    ).to(DEVICE)
    
    # Try multiple paths for robust loading
    base_path = Path(config['pipeline_paths']['best_tcn_lstm_model'])
    spec_path = Path(config['pipeline_paths']['best_specialized_model'])
    
    try:
        model_base.load_state_dict(torch.load(base_path, map_location=DEVICE))
        model_spec.load_state_dict(torch.load(spec_path, map_location=DEVICE))
    except KeyError:
        # Fallback if saved as dict with 'model_state_dict'
        base_dict = torch.load(base_path, map_location=DEVICE)
        spec_dict = torch.load(spec_path, map_location=DEVICE)
        model_base.load_state_dict(base_dict.get('model_state_dict', base_dict))
        model_spec.load_state_dict(spec_dict.get('model_state_dict', spec_dict))
        
    # ── Inference ────────────────────────────────────────────────────────────
    logger.info("Generating Foundation Probabilities...")
    probs_base, targets_aligned = generate_predictions(model_base, loader_base, DEVICE)
    
    logger.info("Generating Specialist Probabilities...")
    probs_spec, _ = generate_predictions(model_spec, loader_spec, DEVICE)
    
    # ── Align Context Features (SeqLen offset) ────────────────────────────────
    logger.info(f"Aligning {len(full_context_df)} context rows to {len(targets_aligned)} predictions...")
    # Because SequenceDataset drops the first `seq_len - 1` targets and the last item
    # y_label = y[idx + seq_len - 1]
    # idx goes from 0 to len(X) - seq_len - 1
    # Thus targets correspond to original index from `seq_len - 1` to `len(X) - 2`
    
    aligned_context_df = full_context_df.iloc[seq_len - 1 : len(full_context_df) - 1].reset_index(drop=True)
    
    # Verify alignment
    if len(aligned_context_df) != len(targets_aligned):
        logger.warning(f"Feature alignment mismatch! Context: {len(aligned_context_df)}, Targets: {len(targets_aligned)}")
        # Truncate to min
        min_len = min(len(aligned_context_df), len(targets_aligned))
        aligned_context_df = aligned_context_df.iloc[:min_len]
        probs_base = probs_base[:min_len]
        probs_spec = probs_spec[:min_len]
        targets_aligned = targets_aligned[:min_len]
        
    # ── Construct Fused Dataset ────────────────────────────────────────────────
    df_fused = aligned_context_df.copy()
    
    df_fused["base_prob_sell"] = probs_base[:, 0]
    df_fused["base_prob_neu"] = probs_base[:, 1]
    df_fused["base_prob_buy"] = probs_base[:, 2]
    
    df_fused["spec_prob_sell"] = probs_spec[:, 0]
    df_fused["spec_prob_neu"] = probs_spec[:, 1]
    df_fused["spec_prob_buy"] = probs_spec[:, 2]
    
    # ── Meta-Labeling Logic ────────────────────────────────────────────────────
    # User Request: Meta_Target = 1 se Pred(Especialista) == Target_Real, senão 0.
    spec_preds_class = np.argmax(probs_spec, axis=1)
    df_fused["meta_target"] = np.where(spec_preds_class == targets_aligned, 1, 0)
    df_fused["true_target"] = targets_aligned
    
    logger.info(f"Fused Dataset Size: {len(df_fused)}. Meta-Target Accuracy: {df_fused['meta_target'].mean():.2%}")
    
    # ── Split Auditor Fused Data ───────────────────────────────────────────────
    split_pct = config['pre_processing']['split']['auditor']['train_ratio']
    split_idx = int(len(df_fused) * split_pct)
    
    train_fused = df_fused.iloc[:split_idx]
    val_fused = df_fused.iloc[split_idx:]
    
    out_path = Path(output_dir)
    train_path = out_path / "train"
    val_path = out_path / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    train_file = train_path / "fused_auditor.parquet"
    val_file = val_path / "fused_auditor.parquet"
    
    pl.DataFrame(train_fused).write_parquet(train_file)
    pl.DataFrame(val_fused).write_parquet(val_file)
    
    logger.info(f"📦 Treino Fused salvo: {train_file} ({len(train_fused)} samples)")
    logger.info(f"📦 Val Fused salvo: {val_file} ({len(val_fused)} samples)")

if __name__ == "__main__":
    setup_logger("auditor_labelling", "")
    conf = load_config()
    
    base_paths = resolve_data_paths({'train_dir': 'AUTO', 'val_dir': 'AUTO'})
    base_train_dir = base_paths[0]
    context_dir = "data/auditor/context/train"
    fused_dir = "data/auditor/dataset_fused"
    
    load_and_predict(conf, base_train_dir, context_dir, fused_dir)
