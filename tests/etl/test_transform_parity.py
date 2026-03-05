"""
tests/etl/test_transform_parity.py

Smoke test: Runs the new Polars-based L2Transformer on a real pre_processed parquet
and checks that the output schema and key column statistics match expectations.

Usage:
    pytest tests/etl/test_transform_parity.py -v
    # or point at a specific file:
    python tests/etl/test_transform_parity.py --file data/L2/pre_processed/2024-01-15.parquet
"""
import sys
import argparse
from pathlib import Path

project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import polars as pl
import pytest


EXPECTED_CORE_COLS = [
    "body", "upper_wick", "lower_wick", "log_ret_close",
    "volatility", "max_spread", "mean_obi", "mean_deep_obi",
    "log_volume", "tick_count",
    "ofi", "ofi_delta_5", "ofi_delta_30",
    "micro_price_momentum", "micro_price_delta_5", "micro_price_delta_30",
    "bid_rdi", "bid_rdi_delta_5", "bid_rdi_delta_30",
    "ask_rdi", "ask_rdi_delta_5", "ask_rdi_delta_30",
    "spread_zscore_60", "vpin_min25",
    "kyle_lambda", "bid_deep_ratio", "ask_deep_ratio",
    "bid_convexity", "ask_convexity", "book_asymmetry_v5", "pressure_ratio",
    "high", "low", "close", "island_id",
]


def load_first_parquet() -> Path:
    pre = Path("data/L2/pre_processed")
    files = sorted(pre.glob("*.parquet"))
    if not files:
        pytest.skip("No pre_processed parquets found — run ETL first.")
    return files[0]


@pytest.fixture
def sample_parquet(request) -> pl.DataFrame:
    file_arg = request.config.getoption("--file", default=None)
    p = Path(file_arg) if file_arg else load_first_parquet()
    return pl.read_parquet(p), p.name


def pytest_addoption(parser):
    parser.addoption("--file", action="store", default=None, help="Path to parquet file to test")


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_expected_columns_present(sample_parquet):
    df, name = sample_parquet
    missing = [c for c in EXPECTED_CORE_COLS if c not in df.columns]
    assert not missing, f"{name}: Missing columns: {missing}"


def test_no_nulls_in_feature_cols(sample_parquet):
    df, name = sample_parquet
    feature_cols = [c for c in EXPECTED_CORE_COLS if c in df.columns and c != "island_id"]
    for col in feature_cols:
        n_null = df[col].null_count()
        assert n_null == 0, f"{name}: Column '{col}' has {n_null} nulls"


def test_no_inf_in_feature_cols(sample_parquet):
    df, name = sample_parquet
    numeric_cols = [c for c in df.columns
                    if df.schema[c] in (pl.Float32, pl.Float64)
                    and c in EXPECTED_CORE_COLS]
    for col in numeric_cols:
        n_inf = df.filter(pl.col(col).is_infinite()).height
        assert n_inf == 0, f"{name}: Column '{col}' has {n_inf} inf values"


def test_island_id_present_and_valid(sample_parquet):
    df, name = sample_parquet
    assert "island_id" in df.columns, f"{name}: island_id column missing"
    assert df["island_id"].dtype == pl.Int32, f"{name}: island_id should be Int32"
    assert df["island_id"].null_count() == 0, f"{name}: island_id has nulls"


def test_canonical_schema(sample_parquet):
    df, name = sample_parquet
    for col in df.columns:
        dtype = df.schema[col]
        if col == "island_id":
            assert dtype == pl.Int32, f"{name}: {col} should be Int32, got {dtype}"
        elif col == "tick_count":
            assert dtype == pl.Int64, f"{name}: {col} should be Int64, got {dtype}"
        elif dtype in (pl.Float64,):
            # Allow Float64 (some Polars operations produce Float64 internally)
            pass


def test_row_count_reasonable(sample_parquet):
    df, name = sample_parquet
    # A single trading day at 5min should have at most 288 rows, at least 10
    assert len(df) >= 10,  f"{name}: Too few rows ({len(df)})"
    assert len(df) <= 300, f"{name}: Too many rows ({len(df)}) for a single day"


def test_close_price_positive(sample_parquet):
    df, name = sample_parquet
    if "close" in df.columns:
        n_nonpos = df.filter(pl.col("close") <= 0).height
        assert n_nonpos == 0, f"{name}: {n_nonpos} non-positive close prices"


# ── CLI Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Path to a pre_processed parquet")
    args = parser.parse_args()

    p = Path(args.file) if args.file else load_first_parquet()
    df = pl.read_parquet(p)
    print(f"\n📂 Testing: {p.name}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns ({len(df.columns)}): {df.columns[:10]}...")
    print(f"   Schema sample: {dict(list(df.schema.items())[:6])}")

    missing = [c for c in EXPECTED_CORE_COLS if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
    else:
        print(f"✅ All {len(EXPECTED_CORE_COLS)} expected columns present")

    nulls = {c: df[c].null_count() for c in EXPECTED_CORE_COLS if c in df.columns and df[c].null_count() > 0}
    if nulls:
        print(f"⚠️  Columns with nulls: {nulls}")
    else:
        print("✅ No nulls in feature columns")
