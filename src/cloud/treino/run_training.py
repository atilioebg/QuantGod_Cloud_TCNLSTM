import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import os
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path to allow importing from src/
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.models.model import QuantGodModel

# Logger setup
log_dir = Path("logs/training")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "fine_tuning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_sequences(X, y, seq_len):
    """Cria sequencias 3D (Batch, Time, Features)"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i+seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)

def train_one_batch_size(batch_size, X_train_t, y_train_t, X_val_t, y_val_t, config, feature_cols, DEVICE):
    """Executa o treino para um tamanho de batch específico."""
    logger.info(f"\n" + "="*50)
    logger.info(f"🚀 INICIANDO TREINO COM BATCH_SIZE: {batch_size}")
    logger.info("="*50)

    seq_len = config['hyperparameters']['seq_len']
    
    model = QuantGodModel(
        num_features=len(feature_cols),
        seq_len=seq_len,
        d_model=config['hyperparameters']['d_model'],
        nhead=config['hyperparameters']['nhead'],
        num_layers=3, 
        num_classes=3,
        dropout=config['hyperparameters']['dropout']
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['lr'])
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = config['hyperparameters']['epochs']
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validação
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for b_X, b_y in val_loader:
                b_X, b_y = b_X.to(DEVICE), b_y.to(DEVICE)
                out = model(b_X)
                pred = torch.argmax(out, dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(b_y.cpu().numpy())
        
        avg_loss = total_loss/len(train_loader)
        f1_w = f1_score(val_targets, val_preds, average='weighted')
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val F1 Weighted: {f1_w:.4f}")
        
    return model

def run_fine_tuning():
    # 1. Load Config
    config_path = Path("src/cloud/treino/training_config.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Parametros solicitados
    BATCH_SIZES = [128, 256, 512]
    
    logger.info("="*50)
    logger.info("🔥 QUANTGOD FINE-TUNING SYSTEM")
    logger.info("="*50)
    logger.info(f"Config File: {config_path}")
    logger.info(f"Data Path: {config['paths']['labelled_dir']}")
    logger.info(f"Batch Sizes to Test: {BATCH_SIZES}")
    logger.info(f"Parameters: {config['hyperparameters']}")
    logger.info("="*50)

    # 2. Load All Labelled Data
    labelled_dir = Path(config['paths']['labelled_dir'])
    parquet_files = sorted(list(labelled_dir.glob("*.parquet")))
    
    if not parquet_files:
        logger.error(f"No labelled data found in {labelled_dir}")
        return

    logger.info(f"Loading {len(parquet_files)} labelled files...")
    
    feature_cols = [
        'body', 'upper_wick', 'lower_wick', 'log_ret_close',
        'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
    ]
    
    dfs = []
    for pf in parquet_files:
        dfs.append(pl.read_parquet(pf, columns=feature_cols + ['target']))
    
    df = pl.concat(dfs)
    logger.info(f"Combined dataset shape: {df.shape}")

    # 3. CHRONOLOGICAL SPLIT LOGIC
    # Para séries temporais, nunca embaralhamos os dados antes do split.
    # Usamos os primeiros 80% para treino e os últimos 20% para validação.
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_raw = df.select('target').to_numpy().flatten().astype(np.int64)
    
    split_idx = int(len(X_raw) * 0.8)
    
    logger.info(f"SPLIT LOGIC: Chronological (80% Train, 20% Val)")
    logger.info(f"Train samples: {split_idx}")
    logger.info(f"Val samples: {len(X_raw) - split_idx}")

    # 4. Normalization (Fit on Train Only)
    scaler = StandardScaler()
    scaler.fit(X_raw[:split_idx])
    X_norm = scaler.transform(X_raw)
    
    # Save Scaler
    scaler_path = Path("data/models/scaler_finetuning.pkl")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    # 5. Sequence Creation
    seq_len = config['hyperparameters']['seq_len']
    logger.info(f"Creating sequences with seq_len={seq_len}...")
    X_seq, y_seq = create_sequences(X_norm, y_raw, seq_len)
    
    # Redefine split index based on sequences
    split_idx_seq = int(len(X_seq) * 0.8)
    X_train, y_train = X_seq[:split_idx_seq], y_seq[:split_idx_seq]
    X_val, y_val = X_seq[split_idx_seq:], y_seq[split_idx_seq:]
    
    # Tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Unified device: {DEVICE}")

    # 6. Fine-Tuning Loop for Batch Sizes
    for b_size in BATCH_SIZES:
        model = train_one_batch_size(b_size, X_train_t, y_train_t, X_val_t, y_val_t, config, feature_cols, DEVICE)
        
        # Save each model variant
        output_name = f"quantgod_finetuned_b{b_size}.pth"
        output_path = Path("data/models") / output_name
        torch.save(model.state_dict(), output_path)
        logger.info(f"✅ Model with Batch Size {b_size} saved to {output_path}")

    logger.info("\n" + "="*50)
    logger.info("🎯 FINE-TUNING COMPLETED SUCCESSFULLY")
    logger.info("="*50)

if __name__ == "__main__":
    run_fine_tuning()
