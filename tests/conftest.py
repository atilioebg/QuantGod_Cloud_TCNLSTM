"""
conftest.py — Shared fixtures and helpers for the QuantGod test suite.

Fixtures are available to all test modules automatically via pytest's
conftest discovery mechanism. No explicit import needed in test files.
"""

import pytest
import numpy as np
from pathlib import Path

# ─── Directory constants (single source of truth for all tests) ──────────────
PRE_PROCESSED_DIR = Path("data/L2/pre_processed")
LABELLED_BASE_DIR = Path("data/L2")
ACTIVE_LABELLED_DIR = LABELLED_BASE_DIR / "labelled_SELL_0004_BUY_0008_1h"

FEATURE_NAMES = [
    "body", "upper_wick", "lower_wick", "log_ret_close",
    "volatility", "max_spread", "mean_obi", "mean_deep_obi", "log_volume",
]
NUM_FEATURES = len(FEATURE_NAMES)   # 9
NUM_CLASSES  = 3                    # SELL=0, NEUTRAL=1, BUY=2
SEQ_LEN      = 720                  # 12h lookback (1-min candles)
META_FEATURES = 14                  # XGBoost auditor input dimension


# ─── Shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_micro_price():
    """Synthetic 720-step micro_price series (positive, trending up slightly)."""
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0.0001, 0.001, 720)
    return np.exp(np.cumsum(log_rets))   # shape (720,), all positive


@pytest.fixture(scope="session")
def sample_probs_balanced():
    """Balanced probability vector summing to 1.0."""
    return np.array([0.3, 0.4, 0.3], dtype=np.float32)


@pytest.fixture(scope="session")
def sample_last_step():
    """Synthetic last-step feature vector shape (9,)."""
    rng = np.random.default_rng(7)
    return rng.standard_normal(NUM_FEATURES).astype(np.float32)


@pytest.fixture(scope="session")
def sample_sequence_batch():
    """
    Synthetic batch of shape (4, 720, 9) — minimum batch to exercise the model
    across multiple examples without touching torch (pure numpy).
    """
    rng = np.random.default_rng(99)
    return rng.standard_normal((4, SEQ_LEN, NUM_FEATURES)).astype(np.float32)