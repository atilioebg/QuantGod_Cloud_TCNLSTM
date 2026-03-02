"""
test_cloud_etl_output.py — Validates output of the pre-processing ETL pipeline.

All feature names and expected column counts are derived dynamically from
master_config.yaml — no hardcodes that can drift from the real pipeline.

Run:
    pytest tests/etl/test_cloud_etl_output.py -v
    PRE_PROCESSED_DIR=data/L2/pre_processed_L2 pytest tests/etl/test_cloud_etl_output.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path

# =============================================================================
# CONFIG — Everything derived from master_config.yaml (v4.9 Gold)
# =============================================================================
_CFG_PATH = Path("src/cloud/base_model/configs/master_config.yaml")

def _load_cfg():
    if _CFG_PATH.exists():
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

_CFG = _load_cfg()
_ETL = _CFG.get("pre_processing", {}).get("etl", {})

# Feature names from master_config (single source of truth)
ALL_FEATURES: list = _CFG.get("model", {}).get("feature_names", [])

# OB levels from config
OB_LEVELS: int = _ETL.get("levels", 200)
OB_COLS: list = []
for i in range(OB_LEVELS):
    OB_COLS.extend([f"bid_{i}_p", f"bid_{i}_s", f"ask_{i}_p", f"ask_{i}_s"])

# Meta columns always written by the ETL pipeline alongside features
_META_COLS = ["close", "open", "high", "low", "ts", "island_id", "tick_count"]

# Total expected: features + meta cols + OB cols
# We use >= check in the test instead of ==, since extra columns (e.g. future OB levels)
# should not fail the test; the important thing is all required columns are PRESENT.
EXPECTED_MIN_COL_COUNT = len(ALL_FEATURES) + len(_META_COLS) + len(OB_COLS)


# =============================================================================
# DIRECTORY RESOLUTION
# =============================================================================
def get_default_dir() -> Path:
    # v4.9: correct default — run_pipeline.py writes to pre_processed_L2
    return Path("data/L2/pre_processed_L2")

TEST_DATA_DIR = Path(os.getenv("PRE_PROCESSED_DIR", get_default_dir()))

def get_test_files() -> list:
    """Returns sorted list of parquet files in the active pre-processed directory."""
    return sorted(list(TEST_DATA_DIR.glob("*.parquet")))


# =============================================================================
# GUARD: Skip entire module if no files found (not an error in CI without data)
# =============================================================================
_TEST_FILES = get_test_files()

if not _TEST_FILES:
    pytest.skip(
        f"No parquet files found in {TEST_DATA_DIR} — skipping ETL output tests.",
        allow_module_level=True,
    )

if not ALL_FEATURES:
    pytest.skip(
        "master_config.yaml not found or 'model.feature_names' is empty.",
        allow_module_level=True,
    )


# =============================================================================
# TEST SUITE
# =============================================================================

@pytest.mark.parametrize("file_path", _TEST_FILES)
class TestCloudDataIntegrity:

    def test_file_exists_and_readable(self, file_path):
        """File must exist and be non-empty."""
        assert file_path.exists(), f"File not found: {file_path}"
        df = pd.read_parquet(file_path)
        assert not df.empty, f"File {file_path.name} is empty."

    def test_all_features_present(self, file_path):
        """All feature_names from master_config must be present as columns."""
        df = pd.read_parquet(file_path)
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        assert not missing, f"Missing features in {file_path.name}: {missing}"

    def test_meta_columns_present(self, file_path):
        """Mandatory meta columns (close, ts, island_id, tick_count) must be present."""
        df = pd.read_parquet(file_path)
        for col in ["close", "ts", "island_id", "tick_count"]:
            assert col in df.columns, f"Meta column '{col}' missing in {file_path.name}"

    def test_orderbook_boundary_columns_present(self, file_path):
        """Check first and last OB level to validate OB depth without full scan."""
        df = pd.read_parquet(file_path)
        max_lvl = OB_LEVELS - 1
        for col in [f"bid_0_p", f"bid_{max_lvl}_p", f"ask_0_p", f"ask_{max_lvl}_p"]:
            assert col in df.columns, f"OB column '{col}' missing in {file_path.name}"

    def test_minimum_column_count(self, file_path):
        """Total columns must be >= (features + meta + OB). Guards against column drops."""
        df = pd.read_parquet(file_path)
        assert len(df.columns) >= EXPECTED_MIN_COL_COUNT, (
            f"{file_path.name}: expected >= {EXPECTED_MIN_COL_COUNT} cols, "
            f"got {len(df.columns)}"
        )

    def test_data_quality_no_nans_or_infs_in_features(self, file_path):
        """Feature columns must have zero NaNs and zero Infs."""
        df = pd.read_parquet(file_path)
        present = [f for f in ALL_FEATURES if f in df.columns]
        nan_counts = df[present].isna().sum().sum()
        assert nan_counts == 0, f"{nan_counts} NaNs in features in {file_path.name}"
        inf_counts = np.isinf(df[present].values).sum()
        assert inf_counts == 0, f"{inf_counts} Infs in features in {file_path.name}"

    def test_orderbook_integrity(self, file_path):
        """Best bid < best ask (no book crossing). Positive close price."""
        df = pd.read_parquet(file_path)
        mask = df["bid_0_p"].notna() & df["ask_0_p"].notna()
        assert (df.loc[mask, "bid_0_p"] < df.loc[mask, "ask_0_p"]).all(), \
            f"Book crossing detected in {file_path.name}"
        assert (df["close"] > 0).all(), f"Non-positive close price in {file_path.name}"

    def test_orderbook_sorted(self, file_path):
        """Sample 10 rows: bids descending, asks ascending (no inversion)."""
        df = pd.read_parquet(file_path)
        sample_idx = np.random.choice(df.index, min(len(df), 10), replace=False)
        for idx in sample_idx:
            row = df.loc[idx]
            bids = [row[f"bid_{i}_p"] for i in range(10) if not np.isnan(row[f"bid_{i}_p"])]
            asks = [row[f"ask_{i}_p"] for i in range(10) if not np.isnan(row[f"ask_{i}_p"])]
            assert all(bids[i] >= bids[i + 1] for i in range(len(bids) - 1)), \
                f"Bids out of order in {file_path.name} row {idx}"
            assert all(asks[i] <= asks[i + 1] for i in range(len(asks) - 1)), \
                f"Asks out of order in {file_path.name} row {idx}"

    def test_chronological_order(self, file_path):
        """Timestamps (ts column or index) must be strictly increasing."""
        df = pd.read_parquet(file_path)
        if "ts" in df.columns:
            assert df["ts"].is_monotonic_increasing, \
                f"Non-monotonic 'ts' in {file_path.name}"
        else:
            assert df.index.is_monotonic_increasing, \
                f"Non-monotonic index in {file_path.name}"

    def test_island_id_no_nulls(self, file_path):
        """island_id must exist and have zero nulls."""
        df = pd.read_parquet(file_path)
        assert "island_id" in df.columns, f"'island_id' missing in {file_path.name}"
        assert df["island_id"].isna().sum() == 0, \
            f"NaN in island_id in {file_path.name}"

    def test_data_types_numeric(self, file_path):
        """All feature columns must be numeric (float or int)."""
        df = pd.read_parquet(file_path)
        for feat in ALL_FEATURES:
            if feat in df.columns:
                assert pd.api.types.is_numeric_dtype(df[feat]), \
                    f"Column '{feat}' in {file_path.name} is not numeric"
