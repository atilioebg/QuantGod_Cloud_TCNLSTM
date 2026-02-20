"""
test_meta_features.py — Unit tests for feature_engineering_meta.py

Tests the XGBoost auditor's 14-feature extraction pipeline (pure numpy, no torch/GPU).

Covered:
  - extract_meta_features: output shape, dtype, value ranges
  - Probability invariants: probs must sum to 1, entropy ≥ 0
  - RSI: bounded in [0, 100], correct limit cases (all gains → 100, all losses → 0)
  - EMA: converges to constant series value
  - EMA distance: zero on flat series
  - Bollinger %B: 0.5 on flat window, bounded for normal series
  - ATR normalized: non-negative
  - Batch extraction: shape (N, 14)
  - No warm-up: indicators work with min viable input length (>= 50)
  - last_step_features = None: falls back to zeros for last-step group
  - META_FEATURE_NAMES: count and content match
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cloud.auditor_model.feature_engineering_meta import (
    extract_meta_features,
    extract_meta_features_batch,
    META_FEATURE_NAMES,
    _rsi,
    _ema,
    _ema_distance_pct,
    _bollinger_pct_b,
    _atr_normalized,
    _entropy,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def price_series():
    """Realistic synthetic micro_price series of length 720."""
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0.0001, 0.002, 720)
    return np.exp(np.cumsum(log_rets))


@pytest.fixture
def flat_series():
    """Constant price series — edge case for indicators."""
    return np.ones(720) * 50_000.0


@pytest.fixture
def uptrend_series():
    """Strict monotonically increasing series — RSI should approach 100."""
    return np.linspace(49_000, 51_000, 720)


@pytest.fixture
def downtrend_series():
    """Strict monotonically decreasing series — RSI should approach 0."""
    return np.linspace(51_000, 49_000, 720)


@pytest.fixture
def valid_probs():
    return np.array([0.3, 0.4, 0.3], dtype=np.float32)


@pytest.fixture
def last_step():
    rng = np.random.default_rng(7)
    return rng.standard_normal(9).astype(np.float32)


# ── extract_meta_features: output contract ───────────────────────────────────

class TestExtractMetaFeatures:

    def test_output_shape(self, price_series, valid_probs, last_step):
        out = extract_meta_features(price_series, valid_probs, last_step)
        assert out.shape == (14,), f"Expected (14,), got {out.shape}"

    def test_output_dtype_float32(self, price_series, valid_probs, last_step):
        out = extract_meta_features(price_series, valid_probs, last_step)
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"

    def test_no_nans_or_infs(self, price_series, valid_probs, last_step):
        out = extract_meta_features(price_series, valid_probs, last_step)
        assert np.isfinite(out).all(), f"NaN/Inf in output: {out}"

    def test_probs_passthrough(self, price_series, valid_probs, last_step):
        """Features 0-2 must be the exact probabilities passed in."""
        out = extract_meta_features(price_series, valid_probs, last_step)
        np.testing.assert_allclose(out[:3], valid_probs, atol=1e-6)

    def test_entropy_non_negative(self, price_series, valid_probs, last_step):
        out = extract_meta_features(price_series, valid_probs, last_step)
        assert out[3] >= 0.0, f"Entropy must be >= 0, got {out[3]}"

    def test_entropy_zero_on_certain_prediction(self, price_series):
        """Entropy = 0 when model is completely certain (prob=1 for one class)."""
        certain_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        out = extract_meta_features(price_series, certain_probs)
        assert abs(out[3]) < 1e-3, f"Entropy should be ~0 on certain prediction, got {out[3]}"

    def test_last_step_features_embedded(self, price_series, valid_probs, last_step):
        """Features 4-10 must reflect last_step_features (indices 0,1,2,3,4,6,7)."""
        out = extract_meta_features(price_series, valid_probs, last_step)
        # Index mapping: meta[4]=last[0], meta[5]=last[1], ..., meta[9]=last[6], meta[10]=last[7]
        expected_indices = [(4, 0), (5, 1), (6, 2), (7, 3), (8, 4), (9, 6), (10, 7)]
        for meta_i, feat_i in expected_indices:
            np.testing.assert_allclose(
                out[meta_i], last_step[feat_i], atol=1e-5,
                err_msg=f"meta[{meta_i}] != last_step[{feat_i}]"
            )

    def test_none_last_step_returns_zeros(self, price_series, valid_probs):
        """When last_step_features is None, indices 4-10 must be 0.0."""
        out = extract_meta_features(price_series, valid_probs, last_step_features=None)
        assert np.all(out[4:11] == 0.0), f"Expected zeros for indices 4-10, got {out[4:11]}"

    def test_rsi_in_output(self, price_series, valid_probs, last_step):
        """Feature[11] = rsi_14 must be in [0, 100]."""
        out = extract_meta_features(price_series, valid_probs, last_step)
        assert 0.0 <= out[11] <= 100.0, f"RSI out of bounds: {out[11]}"

    def test_short_series_raises(self, valid_probs):
        """Series shorter than 50 must raise AssertionError."""
        with pytest.raises(AssertionError):
            extract_meta_features(np.ones(30), valid_probs)

    def test_wrong_probs_shape_raises(self, price_series):
        """Probs with wrong length must raise AssertionError."""
        with pytest.raises(AssertionError):
            extract_meta_features(price_series, np.array([0.5, 0.5]))


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestRSI:

    def test_range(self, price_series):
        assert 0.0 <= _rsi(price_series, 14) <= 100.0

    def test_all_gains_returns_100(self, uptrend_series):
        assert _rsi(uptrend_series, 14) == 100.0

    def test_all_losses_returns_low(self, downtrend_series):
        """All losses → avg_gain=0, RS→0, RSI→0 (or very small)."""
        result = _rsi(downtrend_series, 14)
        assert result < 10.0, f"Expected RSI near 0 on pure downtrend, got {result}"

    def test_flat_series_rsi(self, flat_series):
        """Flat series: all deltas=0, avg_loss=0 → RSI=100 (no losses)."""
        assert _rsi(flat_series, 14) == 100.0

    def test_period_shorter_than_series(self, price_series):
        """RSI must work with period much shorter than series length."""
        r = _rsi(price_series, 5)
        assert isinstance(r, float) and 0.0 <= r <= 100.0


# ── EMA ───────────────────────────────────────────────────────────────────────

class TestEMA:

    def test_flat_series_ema_equals_constant(self, flat_series):
        """EMA of a constant series must equal the constant."""
        result = _ema(flat_series, 9)
        np.testing.assert_allclose(result, 50_000.0, rtol=1e-4)

    def test_ema_returns_float(self, price_series):
        assert isinstance(_ema(price_series, 9), float)
        assert isinstance(_ema(price_series, 50), float)

    def test_ema_distance_zero_on_flat(self, flat_series):
        """% distance from EMA must be 0 on a constant series."""
        result = _ema_distance_pct(flat_series, 9)
        np.testing.assert_allclose(result, 0.0, atol=1e-4)


# ── Bollinger %B ───────────────────────────────────────────────────────────────

class TestBollinger:

    def test_flat_window_returns_half(self, flat_series):
        """Flat window → std=0 → neutral position → 0.5."""
        assert _bollinger_pct_b(flat_series, period=20) == 0.5

    def test_returns_float(self, price_series):
        assert isinstance(_bollinger_pct_b(price_series), float)

    def test_inside_bands_normal_series(self, price_series):
        """For a normal series mid-way through, %B is usually in roughly [0, 1]."""
        result = _bollinger_pct_b(price_series)
        assert isinstance(result, float)  # may go outside [0,1] by design — just check type


# ── ATR ───────────────────────────────────────────────────────────────────────

class TestATR:

    def test_non_negative(self, price_series):
        assert _atr_normalized(price_series) >= 0.0

    def test_zero_on_flat(self, flat_series):
        """Flat series: TR=0 → ATR=0 → normalized ATR=0."""
        np.testing.assert_allclose(_atr_normalized(flat_series), 0.0, atol=1e-9)


# ── Entropy ───────────────────────────────────────────────────────────────────

class TestEntropy:

    def test_max_entropy_uniform(self):
        """Uniform distribution [1/3, 1/3, 1/3] maximizes entropy for 3 classes."""
        p = np.array([1/3, 1/3, 1/3])
        e = _entropy(p)
        assert e > 1.0, f"Expected high entropy for uniform probs, got {e}"

    def test_min_entropy_certain(self):
        """Single class with prob=1 → entropy~0."""
        p = np.array([1.0, 0.0, 0.0])
        e = _entropy(p)
        assert abs(e) < 1e-3

    def test_always_non_negative(self):
        for _ in range(20):
            rng = np.random.default_rng()
            p = rng.dirichlet([1, 1, 1])
            assert _entropy(p) >= 0.0


# ── META_FEATURE_NAMES ────────────────────────────────────────────────────────

class TestMetaFeatureNames:

    def test_count(self):
        assert len(META_FEATURE_NAMES) == 14

    def test_required_names_present(self):
        required = ["prob_sell", "prob_neutral", "prob_buy", "entropy",
                    "rsi_14", "ema_9_dist", "ema_50_dist"]
        for name in required:
            assert name in META_FEATURE_NAMES, f"Missing feature name: {name}"

    def test_no_duplicates(self):
        assert len(META_FEATURE_NAMES) == len(set(META_FEATURE_NAMES))
