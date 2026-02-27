"""
test_window_multiplier.py — Validates that the ETL dynamic window resolution works correctly.

Tests verify that *_min parameters in master_config.yaml are correctly converted to bars
by _load_etl_config and _parse_resample_minutes in transform.py.

The key property being tested:
  bars = window_min / resample_freq_minutes

This ensures every indicator retains its intended temporal meaning regardless of resample_freq.
All floors and expected values are derived dynamically from master_config.yaml — no hardcodes.
"""

import pytest
import yaml
from pathlib import Path

# ── Helper: Load config and compute expected values ─────────────────────────
def load_etl_params():
    """Load active ETL params directly from master_config.yaml."""
    cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    if not cfg_path.exists():
        pytest.skip(f"master_config.yaml not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["pre_processing"]["etl"]


def parse_resample_minutes(freq: str) -> int:
    """Mirror of transform._parse_resample_minutes for test isolation."""
    freq = freq.strip()
    for suffix in ('min', 'T', 'Min'):
        if freq.endswith(suffix):
            return max(1, int(freq.replace(suffix, '')))
    if freq.endswith(('h', 'H')):
        return int(freq[:-1]) * 60
    return 1


# ── Import the live ETL config loader ────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(".").resolve()))
from src.cloud.base_model.pre_processamento.etl.transform import (
    _load_etl_config,
    _parse_resample_minutes as transform_parse_minutes,
    L2Transformer,
)


class TestParseResampleMinutes:
    """Unit tests for the _parse_resample_minutes helper."""

    def test_1min_returns_1(self):
        assert transform_parse_minutes("1min") == 1

    def test_5min_returns_5(self):
        assert transform_parse_minutes("5min") == 5

    def test_15min_returns_15(self):
        assert transform_parse_minutes("15min") == 15

    def test_1h_returns_60(self):
        assert transform_parse_minutes("1h") == 60

    def test_T_alias(self):
        assert transform_parse_minutes("5T") == 5

    def test_unknown_defaults_to_1(self):
        assert transform_parse_minutes("unknown") == 1


class TestWindowMultiplierDynamic:
    """
    Validates that _load_etl_config correctly converts *_min keys to bars.
    All expected values are computed from master_config.yaml — no hardcodes.
    """

    def setup_method(self):
        """Load ground-truth params from the active config."""
        etl = load_etl_params()
        self.resample_freq = etl.get("resample_freq", "5min")
        self.resample_min  = parse_resample_minutes(self.resample_freq)

        self.spread_min    = etl.get("spread_zscore_window_min", 60)
        self.vpin_min      = etl.get("vpin_window_min", 25)
        self.delta_short_m = etl.get("delta_short_min", 5)
        self.delta_long_m  = etl.get("delta_long_min", 30)

        # Expected bars (min-based computation)
        self.expected_spread_bars = max(1, self.spread_min // self.resample_min)
        self.expected_vpin_bars   = max(1, self.vpin_min // self.resample_min)
        self.expected_ds_bars     = max(1, self.delta_short_m // self.resample_min)
        self.expected_dl_bars     = max(1, self.delta_long_m // self.resample_min)

        # Actual resolved config from the loader
        self.cfg = _load_etl_config()

    def test_spread_zscore_window_in_bars(self):
        """spread_zscore_window must equal spread_zscore_window_min / resample_minutes."""
        assert self.cfg["spread_zscore_window"] == self.expected_spread_bars, (
            f"Expected {self.expected_spread_bars} bars for spread_zscore_window "
            f"({self.spread_min}min / {self.resample_min}min), got {self.cfg['spread_zscore_window']}"
        )

    def test_vpin_window_in_bars(self):
        """vpin_window must equal vpin_window_min / resample_minutes."""
        assert self.cfg["vpin_window"] == self.expected_vpin_bars, (
            f"Expected {self.expected_vpin_bars} bars for vpin_window "
            f"({self.vpin_min}min / {self.resample_min}min), got {self.cfg['vpin_window']}"
        )

    def test_delta_short_in_bars(self):
        """delta_short must equal delta_short_min / resample_minutes."""
        assert self.cfg["delta_short"] == self.expected_ds_bars, (
            f"Expected {self.expected_ds_bars} bars for delta_short "
            f"({self.delta_short_m}min / {self.resample_min}min), got {self.cfg['delta_short']}"
        )

    def test_delta_long_in_bars(self):
        """delta_long must equal delta_long_min / resample_minutes."""
        assert self.cfg["delta_long"] == self.expected_dl_bars, (
            f"Expected {self.expected_dl_bars} bars for delta_long "
            f"({self.delta_long_m}min / {self.resample_min}min), got {self.cfg['delta_long']}"
        )

    def test_no_window_is_zero_or_negative(self):
        """Every resolved window must be at least 1 bar (no zero-division or nonsense)."""
        for key in ["spread_zscore_window", "vpin_window", "delta_short", "delta_long"]:
            val = self.cfg[key]
            assert val >= 1, f"Window '{key}' resolved to {val} (must be >= 1)"

    def test_resample_min_stored_correctly(self):
        """resample_min must match the active resample_freq parsed as minutes."""
        assert self.cfg["resample_min"] == self.resample_min

    def test_real_minutes_preserved_in_cfg(self):
        """The *_min keys must be preserved alongside the bar-converted keys."""
        assert self.cfg.get("spread_zscore_window_min") == self.spread_min
        assert self.cfg.get("vpin_window_min") == self.vpin_min
        assert self.cfg.get("delta_short_min") == self.delta_short_m
        assert self.cfg.get("delta_long_min") == self.delta_long_m


class TestL2TransformerWindowInit:
    """Validates that L2Transformer reads bar-resolved windows from _load_etl_config."""

    def test_transformer_uses_resolved_bars(self):
        """L2Transformer internal windows must match bar-resolved values from config."""
        cfg    = _load_etl_config()
        t      = L2Transformer()
        assert t._spread_zscore_window == cfg["spread_zscore_window"]
        assert t._vpin_window          == cfg["vpin_window"]
        assert t._delta_short          == cfg["delta_short"]
        assert t._delta_long           == cfg["delta_long"]

    def test_transformer_real_minute_labels(self):
        """Label attributes must store real minutes for column naming."""
        cfg = _load_etl_config()
        t   = L2Transformer()
        assert t._delta_short_min == cfg["delta_short_min"]
        assert t._delta_long_min  == cfg["delta_long_min"]


class TestLabelledFileRowCount:
    """Dynamic row-count floor: reads resample_freq from config, never hardcodes."""

    def test_floor_derived_from_resample_freq(self):
        """
        Minimum expected rows/day = bars_per_day * 0.8 where
        bars_per_day = 1440 / resample_freq_minutes.
        This test validates the formula itself, not actual parquet files.
        """
        etl           = load_etl_params()
        resample_freq = etl.get("resample_freq", "5min")
        resample_min  = parse_resample_minutes(resample_freq)
        bars_per_day  = 1440 // resample_min
        floor         = int(bars_per_day * 0.8)

        assert floor > 0, "Floor must be positive"
        # At 1min: floor = 1440 * 0.8 = 1152
        # At 5min: floor = 288  * 0.8 = 230
        if resample_min == 1:
            assert floor >= 1000, f"Floor for 1min resample should be >= 1000, got {floor}"
        elif resample_min == 5:
            assert floor >= 200, f"Floor for 5min resample should be >= 200, got {floor}"
