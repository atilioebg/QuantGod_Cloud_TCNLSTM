import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging

# Configure logging at the very top
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Add project root to path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator

def run_gold_tests():
    print("--- Starting Gold Standard Data Integrity Tests v4.6 (Standalone) ---")
    
    # 1. Load Config
    config_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        master_config = yaml.safe_load(f)
    print("OK: Master Config loaded.")

    # 2. Setup Mock Data (24h at 1min freq to match full-day reindex)
    feature_names = master_config['model']['feature_names']
    rows = 1440 # 24h * 60min
    
    print(f"Creating 24h Mock Data ({rows} minutes)...")
    data = {col: np.random.normal(0, 1, rows) for col in feature_names}
    data['micro_price'] = np.linspace(20000, 20100, rows)
    data['obi_l0'] = np.random.uniform(-1, 1, rows)
    data['spread'] = np.random.uniform(0.1, 5.0, rows)
    data['deep_obi_5'] = np.random.uniform(-1, 1, rows)
    data['bid_slope'] = np.random.uniform(10, 100, rows)
    data['ask_slope'] = np.random.uniform(10, 100, rows)
    data['ofi'] = np.random.uniform(-50, 50, rows)
    data['micro_price_momentum'] = np.random.normal(0, 0.001, rows)
    data['pressure_ratio'] = np.random.uniform(-1, 1, rows)
    data['bid_rdi'] = np.random.uniform(0, 5, rows)
    data['ask_rdi'] = np.random.uniform(0, 5, rows)
    data['log_volume'] = np.random.uniform(1, 10, rows)
    data['tick_count'] = np.random.randint(10, 100, rows) # Ensure no natural gaps
    
    data['close'] = data['micro_price']
    data['open']  = data['close'] - 0.5
    data['high']  = data['close'] + 1.0
    data['low']   = data['close'] - 1.0
    
    levels_count = master_config['pre_processing']['etl'].get('levels', 200)
    for i in range(levels_count):
        data[f"bid_{i}_p"] = data['close'] - (i + 1) * 0.1
        data[f"bid_{i}_s"] = np.random.uniform(1, 10, rows)
        data[f"ask_{i}_p"] = data['close'] + (i + 1) * 0.1
        data[f"ask_{i}_s"] = np.random.uniform(1, 10, rows)
    
    df = pd.DataFrame(data)
    df.index = pd.date_range("2023-01-01 00:00:00", periods=rows, freq="1min")
    df['ts'] = (df.index.astype(np.int64) // 10**6)
    
    target_cols = master_config['pre_processing']['etl']['clipping']['target_columns']
    multiplier = master_config['pre_processing']['etl']['clipping']['p99_multiplier']
    
    for col in target_cols:
        if col in df.columns:
            df.loc[df.index[0], col] = 5000.0
            
    print("OK: 24h Mock Data created.")

    # 3. Test Clipping
    print("\n--- Test 1: Clipping Enforcement ---")
    etl_cfg = master_config['pre_processing']['etl']
    transformer = L2Transformer(
        levels=etl_cfg.get('levels', 200),
        sampling_ms=60000, # 1 min sampling for this test
        flow_depth=etl_cfg.get('flow_depth', 5),
        etl_cfg=etl_cfg
    )
    clipped_df = transformer._apply_soft_clipping(df.copy())
    clipping_ok = True
    for col in target_cols:
        if col in clipped_df.columns:
            if clipped_df.loc[clipped_df.index[0], col] >= 5000.0:
                print(f"FAIL: Clipping FAILED for {col}")
                clipping_ok = False
    if clipping_ok: print("PASS: Clipping OK.")

    # 4. Test Lineage
    print("\n--- Test 2: Lineage and Shape ---")
    validator = DataValidator()
    report = validator.validate_integrity(clipped_df, name="Gold Test", feature_list=feature_names)
    lineage_ok = report['shape_integrity']
    if lineage_ok: print("PASS: Lineage and Shape OK.")

    # 5. Test Empty
    print("\n--- Test 3: Empty Dataset ---")
    empty_report = validator.validate_integrity(pd.DataFrame(), name="Empty Test", feature_list=feature_names)
    empty_ok = (not empty_report['is_valid'])
    if empty_ok: print("PASS: Empty Dataset rejected.")

    # 6. Test Level 1 Healing (3 min gap)
    print("\n--- Test 4: Triple Approach Healing (3min Gap) ---")
    gap_df = df.copy()
    transformer.reset_book()
    # Create 3 min gap (from 10:00 to 10:03)
    gap_df.loc[gap_df.index[600:603], 'tick_count'] = 0
    # Also drop them to simulate real missing bars from API
    gap_df_drop = pd.concat([gap_df.iloc[:600], gap_df.iloc[603:]])
    
    healed_df = transformer.apply_feature_engineering(gap_df_drop)
    audit = transformer.audit_report
    # healing_ok if max_gap_after is 0 (or freq) and healed is True
    healing_ok = audit['healed'] is True and audit['max_gap_after'] < 1.0 and audit['max_gap_before'] >= 3.0
    if healing_ok:
        print(f"PASS: 3min Gap Healed. Before: {audit['max_gap_before']}m, After: {audit['max_gap_after']}m")
    else:
        print(f"FAIL: Healing FAILED. Healed: {audit['healed']}, Before: {audit['max_gap_before']}m, After: {audit['max_gap_after']}m")

    # 7. Test Abandonment (65 min gap) - Protocol Island Split v4.6
    print("\n--- Test 5: Abandonment (65min Gap) ---")
    abandon_df_raw = df.copy()
    transformer.reset_book()
    # Create 65 min gap (from 12:00 to 13:05)
    abandon_df_drop = pd.concat([abandon_df_raw.iloc[:720], abandon_df_raw.iloc[785:]])
    
    abandon_df_processed = transformer.apply_feature_engineering(abandon_df_drop)
    # With Island Split, 65min gap should split the day into two islands.
    # Both islands (720 min each) are > 120min, so both should be VALID.
    abandon_report = validator.validate_integrity(abandon_df_processed, name="Abandon Test", feature_list=feature_names)
    
    num_islands = abandon_report.get('num_islands_generated', 0)
    # In Island Split v4.6, a 65min gap should NOT be rejected if valid islands exist
    abandon_ok = (abandon_report['is_valid'] == True and num_islands >= 2)
    if abandon_ok:
        print(f"PASS: 65min Gap handled by Island Split. Islands: {num_islands}.")
    else:
        print(f"FAIL: 65min Gap NOT handled correctly. Valid: {abandon_report['is_valid']}, Islands: {num_islands}")

    if clipping_ok and lineage_ok and empty_ok and healing_ok and abandon_ok:
        print("\nALL GOLD v4.6 (Triple Healing) TESTS PASSED!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_gold_tests()
