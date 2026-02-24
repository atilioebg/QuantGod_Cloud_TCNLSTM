"""
conftest.py — Shared fixtures and helpers for the QuantGod test suite.

Fixtures are available to all test modules automatically via pytest's
conftest discovery mechanism. No explicit import needed in test files.
"""

import pytest
import numpy as np
from pathlib import Path

# ─── Directory constants (single source of truth for all tests) ──────────────
from src.cloud.base_model.utils.experiment_utils import resolve_active_labelled_dir

PRE_PROCESSED_DIR = Path("data/L2/pre_processed")
LABELLED_BASE_DIR = Path("data/L2")
ACTIVE_LABELLED_DIR = resolve_active_labelled_dir()

import yaml

# Load dynamic configs
def load_yaml(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

base_cfg = load_yaml("src/cloud/base_model/configs/base_model_config.yaml")
train_cfg = load_yaml("src/cloud/base_model/treino/training_config.yaml")

# Read dynamically; fallback to defaults if config is not found (e.g. CI without configs)
FEATURE_NAMES = base_cfg.get("model", {}).get("feature_names", [
    "body", "upper_wick", "lower_wick", "log_ret_close",
    "volatility", "max_spread", "mean_obi", "mean_deep_obi", "log_volume",
])
NUM_FEATURES = len(FEATURE_NAMES)
NUM_CLASSES  = base_cfg.get("model", {}).get("num_classes", 3)
SEQ_LEN      = train_cfg.get("hyperparameters", {}).get("seq_len", 720)

from src.cloud.auditor_model.feature_engineering_meta import META_FEATURE_NAMES
META_FEATURES = len(META_FEATURE_NAMES) # Dynamically extract Auditor dimensions


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

# ─── Custom Reporting for Failed Files ───────────────────────────────────────

def pytest_sessionstart(session):
    """Clear the failure report at the start of the session."""
    log_dir = Path("logs/tests")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_file = log_dir / "failed_files_report.log"
    if report_file.exists():
        report_file.unlink()

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Catches test failures and logs the specific file and error to a clean report."""
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call" and rep.failed:
        file_path_str = "Global/Unknown"
        if hasattr(item, "callspec") and "file_path" in item.callspec.params:
            val = item.callspec.params["file_path"]
            file_path_str = getattr(val, "name", str(val))
            
        error_msg = "Unknown Error"
        if call.excinfo:
            # Get the first line of the exception value (the assertion message)
            error_msg = str(call.excinfo.value).split("\n")[0]
            
        log_dir = Path("logs/tests")
        log_dir.mkdir(parents=True, exist_ok=True)
        report_file = log_dir / "failed_files_report.log"
        
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"❌ FILE: {file_path_str} | TEST: {item.name} | ERROR: {error_msg}\n")
