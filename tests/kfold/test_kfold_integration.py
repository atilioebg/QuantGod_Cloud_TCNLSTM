import pytest
import polars as pl
import numpy as np
import pickle
import json
import torch
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(".").resolve()))

from src.cloud.base_model.treino.run_kfold_specialist import run_kfold_specialist

def test_kfold_specialist_execution_and_exports(tmp_path, monkeypatch):
    """
    Integration test: Runs a mini-kfold loop and verifies all artifact exports.
    """
    # 1. Setup Mock Data Environment
    data_dir = tmp_path / "data"
    audit_dir = data_dir / "audit_output" / "splits" / "val"
    audit_dir.mkdir(parents=True)
    
    oof_dir = data_dir / "auditor" / "oof_predictions"
    oof_dir.mkdir(parents=True)
    
    # Create synthetic best_params.json
    config_dir = tmp_path / "src" / "cloud" / "base_model" / "configs"
    config_dir.mkdir(parents=True)
    opt_dir = tmp_path / "src" / "cloud" / "base_model" / "otimizacao"
    opt_dir.mkdir(parents=True)
    
    best_params = {
        "seq_len": 10,
        "tcn_channels": 16,
        "lstm_hidden": 32,
        "num_lstm_layers": 1,
        "dropout": 0.1,
        "lr": 0.001,
        "weight_decay": 0.01,
        "batch_size": 32
    }
    with open(opt_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)
        
    # Create synthetic master_config.yaml
    master_config = {
        "pre_processing": {
            "etl": {"resample_freq": "1min"},
            "labelling": {"sell_threshold": 0.003, "buy_threshold": 0.003, "horizon_minutes": 15},
            "kfold": {
                "n_splits": 2,
                "purge_minutes": 5,
                "specialist_epochs": 1,
                "oof_output_dir": str(oof_dir)
            }
        },
        "training": {
            "specialization_weights": {"class_weights": [1.0, 1.0, 1.0]}
        },
        "optimization": {
            "sniper_weights": {"dir": 0.7, "macro": 0.3},
            "search_space": {"early_stopping_patience": 5}
        },
        "model": {
            "feature_names": ["feat1", "feat2"],
            "num_classes": 3
        },
        "pipeline_paths": {
            "best_tcn_lstm_model": "non_existent_model.pt"
        }
    }
    with open(config_dir / "master_config.yaml", "w") as f:
        yaml.dump(master_config, f) if 'yaml' in locals() else f.write(str(master_config)) # Fallback if yaml not imported

    # Create synthetic validation data
    df = pl.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "target": np.random.randint(0, 3, 100)
    })
    df.write_parquet(audit_dir / "val_chunk.parquet")
    
    # 2. Patch script to use our temp directories
    monkeypatch.chdir(tmp_path)
    # Redirect internal paths via monkeypatch if necessary or rely on chdir
    
    # 3. Execute (1 epoch, 2 folds over 100 rows)
    # We expect some warnings about small dataset, but no crashes.
    try:
        run_kfold_specialist()
    except Exception as e:
        pytest.fail(f"run_kfold_specialist crashed: {e}")
        
    # 4. Verificações (The Core of the User's Question)
    assert (oof_dir / "full_oof.parquet").exists(), "full_oof.parquet was not saved!"
    assert (oof_dir / "scaler_fold_0.pkl").exists(), "scaler_fold_0.pkl was not saved!"
    assert (oof_dir / "scaler_fold_1.pkl").exists(), "scaler_fold_1.pkl was not saved!"
    assert (oof_dir / "fold_0.parquet").exists(), "fold_0.parquet was not saved!"
    assert (oof_dir / "fold_1.parquet").exists(), "fold_1.parquet was not saved!"
    
    # Validate content of full_oof
    oof_df = pl.read_parquet(oof_dir / "full_oof.parquet")
    expected_cols = {"original_row_idx", "spec_prob_sell", "spec_prob_neu", "spec_prob_buy", "spec_pred_class", "true_target", "fold"}
    assert set(oof_df.columns).issubset(expected_cols) or expected_cols.issubset(set(oof_df.columns))
    
    # Check if scalers are valid pickle files
    with open(oof_dir / "scaler_fold_0.pkl", "rb") as f:
        scaler = pickle.load(f)
        assert hasattr(scaler, "mean_"), "Scaler was not properly fitted before saving"

if __name__ == "__main__":
    # Internal minimal import for standalone run
    import yaml
    import pickle
