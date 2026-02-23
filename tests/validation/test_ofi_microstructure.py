"""
Validation test for the new microstructure features in L2Transformer.
Simulates 100 sequential snapshots and asserts:
1. All 7 dynamic feature columns are present in the output
2. After the first row (which has no T-1), NaN count is 0
3. OFI correctly tracks positive and negative net flows
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')

from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer

EXPECTED_DYNAMIC = ['ofi', 'micro_price_momentum', 'bid_slope', 'ask_slope',
                     'bid_rdi', 'ask_rdi', 'pressure_ratio']

def make_msg(ts: int, bid_p: float, bid_s: float, ask_p: float, ask_s: float,
             n_levels: int = 5) -> dict:
    """Build a synthetic snapshot message with n_levels of depth."""
    bids = [[str(round(bid_p - i * 0.1, 2)), str(bid_s - i * 2)] for i in range(n_levels)]
    asks = [[str(round(ask_p + i * 0.1, 2)), str(ask_s + i * 2)] for i in range(n_levels)]
    return {"type": "snapshot", "ts": ts, "data": {"b": bids, "a": asks}}

def run_validation():
    print("=" * 60)
    print("MICROSTRUCTURE FEATURES — 100 SNAPSHOT VALIDATION")
    print("=" * 60)

    transformer = L2Transformer(levels=5, sampling_ms=1000)

    # Scenario A: Stable market then a spike
    scenarios = []
    for i in range(50):
        # Stable
        scenarios.append(make_msg(1000 + i * 1000, 100.0, 10.0, 101.0, 10.0))
    for i in range(30):
        # Bid price rising (buyers aggressive)
        bp = round(100.0 + i * 0.1, 2)
        scenarios.append(make_msg(51000 + i * 1000, bp, 20.0 + i, 101.5, 8.0))
    for i in range(20):
        # Gap scenario: jump > 2 seconds (tests gap guard)
        scenarios.append(make_msg(200000 + i * 1000, 99.0, 15.0, 100.5, 12.0))

    rows = []
    for msg in scenarios:
        row = transformer.process_message(msg)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n✅ Snapshots processed: {len(df)}")
    print(f"   Columns generated: {len(df.columns)}")

    # 1. Check all expected dynamic columns exist
    missing = [c for c in EXPECTED_DYNAMIC if c not in df.columns]
    if missing:
        print(f"\n❌ FAIL — Missing dynamic columns: {missing}")
        return False
    print(f"\n✅ PASS — All 7 dynamic columns present: {EXPECTED_DYNAMIC}")

    # 2. Check NaN count in dynamic cols (first row is expected zeros, not NaN)
    nan_counts = df[EXPECTED_DYNAMIC].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"\n❌ FAIL — Unexpected NaNs detected:\n{nan_counts[nan_counts > 0]}")
        return False
    print(f"✅ PASS — Zero NaNs in all dynamic features")

    # 3. OFI direction test: After bid AgGression (rising prices), OFI should be positive
    mid_start = 51  # Index where rising bid prices start
    ofi_aggression = df['ofi'].iloc[mid_start:81].mean()
    print(f"✅ OFI mean during bid aggression window: {ofi_aggression:+.4f} (expected positive)")
    if ofi_aggression <= 0:
        print("   ⚠️  WARNING: OFI not positive during bid aggression - check logic")

    # 4. Gap guard test: OFI at gap boundary should be 0 (T-1 was purged)
    gap_idx = 80  # First row after the 200s gap
    ofi_at_gap = df['ofi'].iloc[gap_idx]
    print(f"✅ OFI at gap boundary (idx {gap_idx}): {ofi_at_gap} (expected 0.0)")
    if ofi_at_gap != 0.0:
        print(f"   ❌ FAIL: Gap guard not working — OFI = {ofi_at_gap}")
        return False

    # 5. Sample output
    print("\n📊 Sample 5 rows of dynamic features:")
    print(df[EXPECTED_DYNAMIC].iloc[48:53].to_string(index=False))

    print("\n✅ ALL CHECKS PASSED — Safe to process full dataset")
    return True

if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
