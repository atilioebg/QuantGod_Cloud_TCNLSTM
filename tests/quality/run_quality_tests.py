import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Add project root to path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator

def run_gold_tests():
    print("🚀 Starting Gold Standard Data Integrity Tests (Standalone)...")
    
    # 1. Load Config
    config_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        master_config = yaml.safe_load(f)
    print("✅ Master Config loaded.")

    # 2. Setup Mock Data
    feature_names = master_config['model']['feature_names']
    rows = 1000
    data = {col: np.random.normal(0, 1, rows) for col in feature_names}
    data['close'] = np.linspace(20000, 21000, rows)
    data['open'] = data['close'] - 10
    data['high'] = data['close'] + 5
    data['low'] = data['close'] - 15
    data['log_volume'] = np.random.normal(10, 1, rows)
    
    df = pd.DataFrame(data)
    df.index = pd.date_range("2023-01-01", periods=rows, freq="1min")
    
    target_cols = master_config['pre_processing']['etl']['clipping']['target_columns']
    multiplier = master_config['pre_processing']['etl']['clipping']['p99_multiplier']
    
    for col in target_cols:
        if col in df.columns:
            df.loc[df.index[0], col] = 5000.0 # Extreme outlier
    print("✅ Mock Data with 5000x outliers created.")

    # 3. Test Clipping
    print("\n--- Test 1: Clipping Enforcement ---")
    # L2Transformer expects exactly these arguments based on outline
    etl_cfg = master_config['pre_processing']['etl']
    transformer = L2Transformer(
        levels=etl_cfg.get('levels', 200),
        sampling_ms=etl_cfg.get('sampling_ms', 1000),
        flow_depth=etl_cfg.get('flow_depth', 5),
        etl_cfg=etl_cfg
    )
    clipped_df = transformer._apply_soft_clipping(df.copy())
    
    failed = False
    for col in target_cols:
        if col in clipped_df.columns:
            p99 = df[col].quantile(0.99)
            limit = p99 * multiplier * 1.05
            actual_max = clipped_df[col].max()
            if actual_max > limit:
                print(f"❌ Feature {col} FAILED clipping: max {actual_max:.4f} > limit {limit:.4f}")
                failed = True
            else:
                print(f"✅ Feature {col} PASSED clipping: max {actual_max:.4f} <= limit {limit:.4f}")
    
    # 4. Test DataValidator (Gold Level)
    print("\n--- Test 2: Gold DataValidator ---")
    validator = DataValidator()
    # Should log Gold Validation success but also warnings for high tails in other columns if any
    is_valid = validator.validate_integrity(clipped_df, name="Gold Test (Clipped)")
    if is_valid:
        print("✅ DataValidator PASSED (Clipped dataset is clean).")
    else:
        print("❌ DataValidator FAILED (Found NaNs/Infs).")

    if not failed and is_valid:
        print("\n✨ ALL GOLD STANDARD TESTS PASSED! ✨")
        sys.exit(0)
    else:
        print("\n⚠️ SOME TESTS FAILED. CHECK LOGS. ⚠️")
        sys.exit(1)

if __name__ == "__main__":
    run_gold_tests()
