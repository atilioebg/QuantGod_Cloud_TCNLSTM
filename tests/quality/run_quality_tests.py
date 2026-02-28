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
    print("🚀 Starting Gold Standard Data Integrity Tests v4.6 (Standalone)...")
    
    # 1. Load Config
    config_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        master_config = yaml.safe_load(f)
    print("✅ Master Config loaded.")

    # 2. Setup Mock Data
    feature_names = master_config['model']['feature_names']
    rows = 1000
    
    # Generate features from config
    data = {col: np.random.normal(0, 1, rows) for col in feature_names}
    
    # Add price columns (Mandatory for validator)
    # Note: open, high, low ARE NOT usually in feature_names (which has body, wicks, etc)
    data['close'] = np.linspace(20000, 21000, rows)
    data['open']  = data['close'] - 10
    data['high']  = data['close'] + 5
    data['low']   = data['close'] - 15
    # log_volume is already in feature_names
    
    df = pd.DataFrame(data)
    df.index = pd.date_range("2023-01-01", periods=rows, freq="1min")
    
    # Metrics count = 30 (from config) + 3 (open, high, low) = 33
    
    target_cols = master_config['pre_processing']['etl']['clipping']['target_columns']
    multiplier = master_config['pre_processing']['etl']['clipping']['p99_multiplier']
    
    for col in target_cols:
        if col in df.columns:
            df.loc[df.index[0], col] = 5000.0 # Extreme outlier
    print(f"✅ Mock Data with {len(feature_names)} features + 3 Price Candles + 1 Close created.")

    # 3. Test Clipping
    print("\n--- Test 1: Clipping Enforcement ---")
    etl_cfg = master_config['pre_processing']['etl']
    transformer = L2Transformer(
        levels=etl_cfg.get('levels', 200),
        sampling_ms=etl_cfg.get('sampling_ms', 1000),
        flow_depth=etl_cfg.get('flow_depth', 5),
        etl_cfg=etl_cfg
    )
    clipped_df = transformer._apply_soft_clipping(df.copy())
    
    clipping_ok = True
    for col in target_cols:
        if col in clipped_df.columns:
            p99 = df[col].quantile(0.99)
            limit = p99 * multiplier * 1.1
            actual_max = clipped_df[col].max()
            if actual_max > limit:
                print(f"❌ Feature {col} FAILED clipping: max {actual_max:.4f} > limit {limit:.4f}")
                clipping_ok = False
    
    # 4. Test DataValidator (Lineage & Shape)
    print("\n--- Test 2: Gold DataValidator (Lineage & Shape) ---")
    validator = DataValidator()
    
    # Inject a mock logit to test [XGB_ONLY] lineage labeling
    clipped_df['base_logit_buy'] = 0.5
    
    # Define a custom feature list for the test that includes the mock logit
    test_features = feature_names + ['base_logit_buy']
    health_report = validator.validate_integrity(clipped_df, name="Gold Test (Lineage)", feature_list=test_features)
    
    lineage_and_shape_ok = health_report['is_valid']
    if lineage_and_shape_ok:
        print("✅ DataValidator PASSED (Lineage labeling and Shape match).")
    else:
        print("❌ DataValidator FAILED (Check logs).")

    # 5. Test Shape Failure
    print("\n--- Test 3: Architecture Integrity (Shape Failure) ---")
    # Intentional mismatch: we tell validator to expect Only the original 30 features, but we have 31 metrics
    bad_report = validator.validate_integrity(clipped_df, name="Shape Test (Failure)", feature_list=feature_names)
    shape_detection_ok = (not bad_report['shape_integrity'] and not bad_report['is_valid'])
    
    if shape_detection_ok:
        print("✅ Shape Integrity PASSED (Correctly detected mismatch).")

    # 6. Test Empty Dataset Failure
    print("\n--- Test 4: Empty Dataset Rejection ---")
    empty_df = pd.DataFrame()
    empty_report = validator.validate_integrity(empty_df, name="Empty Test", feature_list=feature_names)
    empty_rejection_ok = (not empty_report['is_valid'])
    
    if empty_rejection_ok:
        print("✅ Empty Dataset Rejection PASSED (Correctly marked as INVALID).")

    if clipping_ok and lineage_and_shape_ok and shape_detection_ok and empty_rejection_ok:
        print("\n✨ ALL GOLD STANDARD v4.6 TESTS PASSED! ✨")
        sys.exit(0)
    else:
        print(f"\n⚠️ TESTS FAILED: Clipping={clipping_ok}, Lineage/Shape={lineage_and_shape_ok}, ShapeDetection={shape_detection_ok}, Empty={empty_rejection_ok}")
        sys.exit(1)

if __name__ == "__main__":
    run_gold_tests()
