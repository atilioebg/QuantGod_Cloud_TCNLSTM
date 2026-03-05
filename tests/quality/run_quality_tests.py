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

    # 2. Setup Mock Data (24h at 30s freq to ensure high density)
    feature_names = master_config['model']['feature_names']
    rows = 2880 # 24h * 60min * 2
    
    # Identify resample_freq
    etl_cfg = master_config['pre_processing']['etl']
    resample_freq = etl_cfg.get('resample_freq', '1min')
    resample_min = int(pd.to_timedelta(resample_freq).total_seconds() // 60)
    
    print(f"Creating 24h Mock Data ({rows} snapshots) for {resample_freq} resolution...")
    data = {col: np.random.normal(0, 1, rows) for col in feature_names}
    data['micro_price'] = np.linspace(20000, 20100, rows)
    
    # Fill required base columns
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
    data['tick_count'] = np.random.randint(10, 100, rows)
    
    data['close'] = data['micro_price']
    data['open']  = data['close'] - 0.5
    data['high']  = data['close'] + 1.0
    data['low']   = data['close'] - 1.0
    
    levels_count = etl_cfg.get('levels', 200)
    for i in range(levels_count):
        data[f"bid_{i}_p"] = data['close'] - (i + 1) * 0.1
        data[f"bid_{i}_s"] = np.random.uniform(1, 10, rows)
        data[f"ask_{i}_p"] = data['close'] + (i + 1) * 0.1
        data[f"ask_{i}_s"] = np.random.uniform(1, 10, rows)
    
    df = pd.DataFrame(data)
    # Start at 00:00:00 UTC to align with Day Anchor
    df.index = pd.date_range("2023-01-01 00:00:00", periods=rows, freq="30s", tz='UTC')
    df['ts'] = (df.index.values.astype('datetime64[ms]').astype(np.int64))
    df['island_id'] = 0 # Mandatory for v4.6+
    
    target_cols = etl_cfg['clipping']['target_columns']
    for col in target_cols:
        if col in df.columns:
            df.loc[df.index[0], col] = 5000.0
            
    print("OK: 24h Mock Data created.")

    # 3. Test Clipping
    print("\n--- Test 1: Clipping Enforcement ---")
    transformer = L2Transformer(
        levels=levels_count,
        sampling_ms=1000, # Realistic sampling
        etl_cfg=etl_cfg
    )
    import polars as pl
    pl_df = pl.from_pandas(df)
    clipped_pl = transformer._apply_soft_clipping_pl(pl_df)
    clipping_ok = True
    for col in target_cols:
        if col in clipped_pl.columns:
            # Check the first row (where we injected 5000.0)
            val = clipped_pl.select(pl.col(col)).row(0)[0]
            if val >= 5000.0:
                print(f"FAIL: Clipping FAILED for {col} (val={val})")
                clipping_ok = False
    if clipping_ok: print("PASS: Clipping OK.")
    # 4. Test Lineage
    print("\n--- Test 2: Lineage and Shape ---")
    validator = DataValidator()
    delta_short_min = etl_cfg.get('delta_short_min', 5)
    
    # Convert Polars clipped data back to Pandas for legacy validation step
    clipped_df = clipped_pl.to_pandas()
    if "datetime" in clipped_df.columns:
        clipped_df = clipped_df.set_index("datetime")
        
    report = validator.validate_integrity(
        clipped_df,
        name="Gold Test",
        feature_list=feature_names,
        resample_freq=resample_freq,
        delta_short_min=delta_short_min
    )
    lineage_ok = report['shape_integrity']
    if lineage_ok: print("PASS: Lineage and Shape OK.")

    # 5. Test Empty
    print("\n--- Test 3: Empty Dataset ---")
    empty_report = validator.validate_integrity(pd.DataFrame(), name="Empty Test", feature_list=feature_names)
    empty_ok = (not empty_report['is_valid'])
    if empty_ok: print("PASS: Empty Dataset rejected.")

    # 6. Test Level 1 Healing (Adaptive Gap)
    # Gap size should be enough to create empty bars regardless of resolution
    gap_min = max(3.0, resample_min + 1.0)
    print(f"\n--- Test 4: Triple Approach Healing ({gap_min}min Gap) ---")
    gap_df = df.copy()
    transformer.reset_book()
    
    # Create gap (gap_min converted to snapshots)
    gap_snaps = int(gap_min * 2)
    start_idx = 1200
    gap_df_drop = pd.concat([gap_df.iloc[:start_idx], gap_df.iloc[start_idx + gap_snaps:]])
    
    healed_pl = transformer.apply_feature_engineering(gap_df_drop)
    # Convert back to pandas for the legacy test assertions
    healed_df = healed_pl.to_pandas()
    if "datetime" in healed_df.columns:
        healed_df = healed_df.set_index("datetime")
        
    print(f"Test 4 Shape: {healed_df.shape}")
    print(f"Test 4 Index: {healed_df.index[0]} to {healed_df.index[-1]}")
    audit = transformer.audit_report
    print(f"Audit Report Max Gap Before: {audit['max_gap_before']}")

    # healing_ok if max_gap_after is 0 (or freq) and healed is True
    # If resample_freq is large (e.g. 5min), max_gap_before might pick up the reindexed grid gaps
    healing_ok = audit['healed'] is True and audit['max_gap_after'] < resample_min
    if healing_ok:
        print(f"PASS: {gap_min}min Gap Healed. Before: {audit['max_gap_before']}m, After: {audit['max_gap_after']}m")
    else:
        print(f"FAIL: Healing FAILED. Healed: {audit['healed']}, Before: {audit['max_gap_before']}m, After: {audit['max_gap_after']}m")

    # 7. Test Abandonment (65 min gap) - Protocol Island Split v4.6
    print("\n--- Test 5: Abandonment (65min Gap) ---")
    abandon_df_raw = df.copy()
    transformer.reset_book()
    # Create 65 min gap (130 snapshots)
    abandon_df_drop = pd.concat([abandon_df_raw.iloc[:1440], abandon_df_raw.iloc[1570:]])
    
    abandon_pl = transformer.apply_feature_engineering(abandon_df_drop)
    # Convert back to pandas for the legacy test assertions/validator
    abandon_df_processed = abandon_pl.to_pandas()
    if "datetime" in abandon_df_processed.columns:
        abandon_df_processed = abandon_df_processed.set_index("datetime")

    print(f"Abandon Test Shape: {abandon_df_processed.shape}")
    print(f"Abandon Test Index: {abandon_df_processed.index[0]} to {abandon_df_processed.index[-1]}")
    print(f"Audit Report Max Gap Before: {transformer.audit_report['max_gap_before']}")

    abandon_report = validator.validate_integrity(
        abandon_df_processed,
        name="Abandon Test",
        feature_list=feature_names,
        resample_freq=resample_freq,
        delta_short_min=delta_short_min
    )
    
    num_islands = abandon_report.get('num_islands_generated', 0)
    valid_islands = len(abandon_report.get('valid_island_ids', []))
    
    # We expect BOTH islands to survive (>120min each) and is_valid=True.
    abandon_ok = (abandon_report['is_valid'] == True and valid_islands >= 2)
    
    if abandon_ok:
        print(f"PASS: 65min Gap handled by Island Split. Islands: {num_islands}, Survived: {valid_islands}.")
    else:
        print(f"FAIL: 65min Gap NOT handled correctly. Valid: {abandon_report['is_valid']}, Islands: {num_islands}, Survived: {valid_islands}")

    if clipping_ok and lineage_ok and empty_ok and healing_ok and abandon_ok:
        print("\nALL GOLD v4.6 (Adaptive) TESTS PASSED!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_gold_tests()
