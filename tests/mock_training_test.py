import torch
import polars as pl
import numpy as np
import yaml
from pathlib import Path
import sys
import os

# Ajustar path para importar do projeto
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.treino.run_specialization import QuantGodLazyDataset, SpecialistObjective
import optuna

def create_mock_data(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Lista de features baseada no master_config.yaml (simplificada para o teste)
    feature_cols = [
        "body", "upper_wick", "lower_wick", "log_ret_close", "volatility", 
        "max_spread", "mean_obi", "mean_deep_obi", "log_volume", "ofi"
    ]
    # Completando até as 30 features se necessário, ou apenas usando uma lista menor para o mock dataset
    # O modelo TCN/LSTM precisará de num_features exato. Vamos usar 30.
    all_features = feature_cols + [f"feat_{i}" for i in range(20)]
    
    # Criar 2 arquivos mock
    for i in range(2):
        n_rows = 1000
        data = {f: np.random.randn(n_rows).astype(np.float32) for f in all_features}
        data['target'] = np.random.choice([0, 1, 2], n_rows).astype(np.int64)
        data['island_id'] = np.repeat([i*10 + 1, i*10 + 2], n_rows//2).astype(np.int64)
        
        df = pl.DataFrame(data)
        df.write_parquet(data_dir / f"mock_day_{i}.parquet")
    
    return all_features

def test_mock_pipeline():
    print("🚀 Iniciando Teste Mockado do Pipeline de Treino...")
    
    mock_dir = Path("tmp/mock_labelled")
    feature_names = create_mock_data(mock_dir / "train")
    _ = create_mock_data(mock_dir / "val")
    
    # Mock Config
    config = {
        'model': {'feature_names': feature_names},
        'optimization': {
            'search_space': {
                'epochs': 2,
                'early_stopping_patience': 1,
                'batch_size': [16, 32, 64],
                'seq_len': [10, 20, 30],
                'lr': [1e-4, 1e-3, 1e-2],
                'weight_decay': [1e-5, 1e-4, 1e-3],
                'dropout': [0.1, 0.2, 0.3],
                'tcn_channels': [[16, 16]],
                'lstm_hidden': [32],
                'num_lstm_layers': [1]
            },
            'sniper_weights': {'dir': 0.7, 'macro': 0.3}
        },
        'training': {
            'specialization_weights': {
                'spec_use_auto_class_weights': True,
                'spec_optimize_gamma': False,
                'spec_optimize_smoothing': False
            }
        },
        'training_optimization': {
            'use_amp': True,
            'use_gradient_accumulation': True,
            'accumulation_steps': 2,
            'use_epoch_chunking': True,
            'samples_per_epoch': 100,
            'use_class_subsampling': True,
            'subsample_class_target': 1,
            'subsample_keep_ratio': 0.5
        }
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    train_files = list((mock_dir / "train").glob("*.parquet"))
    val_files = list((mock_dir / "val").glob("*.parquet"))

    # 1. Test Dataset
    print("Testing QuantGodLazyDataset...")
    dataset = QuantGodLazyDataset(train_files, feature_names, seq_len=10, config=config, is_train=True)
    print(f"Dataset Len (with chunking): {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample X shape: {x.shape}, Y: {y}")
    assert x.shape == (10, 30)

    # 2. Test Objective
    print("Testing SpecialistObjective loop...")
    spec_space = {
        'batch_size': [16, 64], 'seq_len': [10, 20], 'lr': [1e-3], 
        'weight_decay': [1e-4], 'dropout': [0.1],
        'tcn_channels': [16, 32], 'lstm_hidden': [32], 'num_lstm_layers': [1],
        'spec_loss_gamma': [2.0], 'spec_loss_smoothing': [0.1]
    }
    
    objective = SpecialistObjective(
        config=config,
        spec_space=spec_space,
        train_files=train_files,
        val_files=val_files,
        class_weights=[1.0, 1.0, 1.0],
        DEVICE=DEVICE,
        best_params={}
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    
    print(f"✅ Mock Test COMPLETED. Best Score: {study.best_value}")

if __name__ == "__main__":
    try:
        test_mock_pipeline()
    finally:
        # Cleanup
        import shutil
        if os.path.exists("tmp/mock_labelled"):
            shutil.rmtree("tmp/mock_labelled")
