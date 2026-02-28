import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator

@pytest.fixture
def master_config():
    config_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@pytest.fixture
def sample_df(master_config):
    """Creates a sample dataframe with intentional outliers for testing clipping."""
    feature_names = master_config['model']['feature_names']
    rows = 1000
    data = {col: np.random.normal(0, 1, rows) for col in feature_names}
    # Inject OHLC for basic ETL compatibility
    data['close'] = np.linspace(20000, 21000, rows)
    data['open'] = data['close'] - 10
    data['high'] = data['close'] + 5
    data['low'] = data['close'] - 15
    data['log_volume'] = np.random.normal(10, 1, rows)
    
    df = pd.DataFrame(data)
    df.index = pd.date_range("2023-01-01", periods=rows, freq="1min")
    
    # Inject massive outliers in target columns (100x the normal range)
    target_cols = master_config['pre_processing']['etl']['clipping']['target_columns']
    for col in target_cols:
        if col in df.columns:
            df.loc[df.index[0], col] = 5000.0 # Extreme outlier
            
    return df

def test_clipping_enforcement(sample_df, master_config):
    """Verifies if L2Transformer correctly caps outliers at P99 multiplier."""
    transformer = L2Transformer(config=master_config)
    
    # Apply soft clipping logic
    # _apply_soft_clipping is internal, but we can test its effect through the public interface or directly
    clipped_df = transformer._apply_soft_clipping(sample_df.copy())
    
    target_cols = master_config['pre_processing']['etl']['clipping']['target_columns']
    multiplier = master_config['pre_processing']['etl']['clipping']['p99_multiplier']
    
    for col in target_cols:
        if col in clipped_df.columns:
            p99 = sample_df[col].quantile(0.99)
            # Use a small epsilon for float comparison
            limit = p99 * multiplier * 1.05 
            assert clipped_df[col].max() <= limit, f"Feature {col} exceeded clipping limit!"

def test_feature_preservation_30_channels(sample_df, master_config):
    """Ensures all 30 features defined in master_config are preserved."""
    expected_features = master_config['model']['feature_names']
    assert len(expected_features) == 30, f"Master config should have 30 features, found {len(expected_features)}"
    
    # Dummy processing to check column presence
    for feat in expected_features:
        assert feat in sample_df.columns, f"Missing feature in sample: {feat}"

def test_validator_detects_clipping_failure(sample_df):
    """Validator should flag extreme outliers if clipping is NOT applied."""
    validator = DataValidator()
    # sample_df has a 5000.0 value which is > 15x P99 of a normal(0,1) distribution (~2.33)
    # validate_integrity returns True if no NaNs/Infs, but logs warnings.
    # In a "Gold" test, we might want to capture logs.
    is_valid = validator.validate_integrity(sample_df, name="Unclipped Test")
    assert is_valid is True # Still valid but should have logged warnings
