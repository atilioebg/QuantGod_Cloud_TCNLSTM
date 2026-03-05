"""
test_config_integrity.py — Validates that all YAML configuration files are
consistent, complete, and internally coherent.

No data or models are loaded — pure config file validation.

Covered:
  - master_config.yaml: unified configuration file encompassing paths, pre_processing, model architecture, training, and optimization boundaries.
  - auditor_config.yaml: walk-forward folds, XGBoost params, path keys
  - Cross-config consistency check where auditor keys align with master ones.
"""

import pytest
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config paths ──────────────────────────────────────────────────────────────
MASTER_CFG    = Path("src/cloud/base_model/configs/master_config.yaml")
AUDITOR_CFG   = Path("src/cloud/auditor_model/configs/auditor_config.yaml")

def load(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── master_config.yaml ────────────────────────────────────────────────────────
class TestMasterConfig:

    def test_file_exists(self):
        assert MASTER_CFG.exists(), f"Config not found: {MASTER_CFG}"

    def test_top_level_keys(self):
        cfg = load(MASTER_CFG)
        for key in ["pipeline_paths", "pre_processing", "model", "training", "optimization"]:
            assert key in cfg, f"Missing key '{key}'"

    def test_feature_names_no_duplicates(self):
        cfg = load(MASTER_CFG)
        names = cfg["model"]["feature_names"]
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_num_classes_matches_weights(self):
        cfg = load(MASTER_CFG)
        num_c = cfg["model"]["num_classes"]
        w_found = cfg["training"]["foundation_weights"]["class_weights"]
        w_spec  = cfg["training"]["specialization_weights"]["class_weights"]
        assert len(w_found) == num_c, f"Expected {num_c} base foundation_weights, got {len(w_found)}"
        assert len(w_spec) == num_c, f"Expected {num_c} specialized weights, got {len(w_spec)}"

    def test_split_ratios_exist(self):
        cfg = load(MASTER_CFG)
        assert "split" in cfg["pre_processing"]
        assert "base" in cfg["pre_processing"]["split"]
        assert "specialized" in cfg["pre_processing"]["split"]
        
    def test_focal_loss_params(self):
        cfg = load(MASTER_CFG)
        assert "gamma" in cfg["training"]["foundation_weights"]
        assert "smoothing" in cfg["training"]["specialization_weights"]

    def test_optimization_metrics_exist(self):
        cfg = load(MASTER_CFG)
        assert "base_metric" in cfg["optimization"]
        assert "specialized_metric" in cfg["optimization"]

    def test_batch_size_safe(self):
        cfg = load(MASTER_CFG)
        bs_list = cfg["optimization"]["search_space"]["batch_size"]
        assert isinstance(bs_list, list), "batch_size deve ser uma lista no search_space"
        assert max(bs_list) <= 2048, f"O batch_size máximo ({max(bs_list)}) excede o limite de segurança para evitar OOM"

# ── Auditor config (consolidated into master_config.yaml since v4.5) ──────────
class TestAuditorConfig:
    """
    The auditor_config.yaml was consolidated into master_config.yaml (v4.5+).
    These tests now validate the auditor sections within master_config.yaml:
      - model.auditor  → XGBoost params + thresholds
      - pipeline_paths → auditor model + scaler paths
      - pre_processing.split.auditor → train/val ratios
      - pre_processing.kfold        → n_splits (walk-forward folds)
    """

    def test_master_config_exists(self):
        assert MASTER_CFG.exists(), f"master_config.yaml not found: {MASTER_CFG}"

    def test_required_auditor_keys(self):
        cfg = load(MASTER_CFG)
        assert "auditor" in cfg["model"], "model.auditor section missing"
        auditor = cfg["model"]["auditor"]
        for key in ["num_features", "num_classes", "params"]:
            assert key in auditor, f"model.auditor.{key} missing"

    def test_xgb_output_is_json(self):
        cfg = load(MASTER_CFG)
        xgb_path = cfg["pipeline_paths"]["auditor_model"]
        assert xgb_path.endswith(".json"), \
            f"auditor_model path should end in .json, got: {xgb_path}"

    def test_n_folds_minimum(self):
        """Walk-forward requires at least 3 folds to be statistically meaningful."""
        cfg = load(MASTER_CFG)
        n_folds = cfg["pre_processing"]["kfold"]["n_splits"]
        assert n_folds >= 3, \
            f"kfold.n_splits={n_folds} — minimum 3 for valid walk-forward"

    def test_base_model_checkpoint_is_pt(self):
        cfg = load(MASTER_CFG)
        checkpoint = cfg["pipeline_paths"]["best_tcn_lstm_model"]
        assert checkpoint.endswith(".pt"), \
            f"best_tcn_lstm_model should end in .pt, got: {checkpoint}"

    def test_auditor_split_ratios_sum_to_one(self):
        cfg = load(MASTER_CFG)
        auditor_split = cfg["pre_processing"]["split"]["auditor"]
        total = auditor_split["train_ratio"] + auditor_split["val_ratio"]
        assert abs(total - 1.0) < 1e-9, \
            f"auditor split ratios must sum to 1.0, got {total}"

    def test_auditor_xgb_params_present(self):
        cfg = load(MASTER_CFG)
        params = cfg["model"]["auditor"]["params"]
        for key in ["n_estimators", "max_depth", "learning_rate"]:
            assert key in params, f"model.auditor.params.{key} missing"
