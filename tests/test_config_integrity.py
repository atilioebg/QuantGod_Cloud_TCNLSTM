"""
test_config_integrity.py — Validates that all YAML configuration files are
consistent, complete, and internally coherent.

No data or models are loaded — pure config file validation.

Covered:
  - base_model_config.yaml: feature names, class weights, seq_len, optimizer, scheduler
  - training_config.yaml: paths exist as keys, hyperparameters within valid ranges
  - optimization_config.yaml: search space types, batch_size safety cap
  - auditor_config.yaml: walk-forward folds, XGBoost params, path keys
  - Cross-config consistency: labelled_dir matches between training and auditor configs
  - No duplicate feature names in base_model_config
"""

import pytest
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config paths ──────────────────────────────────────────────────────────────
BASE_CFG      = Path("src/cloud/base_model/configs/base_model_config.yaml")
TRAIN_CFG     = Path("src/cloud/base_model/treino/training_config.yaml")
OPT_CFG       = Path("src/cloud/base_model/otimizacao/optimization_config.yaml")
AUDITOR_CFG   = Path("src/cloud/auditor_model/configs/auditor_config.yaml")

def load(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── base_model_config.yaml ────────────────────────────────────────────────────

class TestBaseModelConfig:

    def test_file_exists(self):
        assert BASE_CFG.exists(), f"Config not found: {BASE_CFG}"

    def test_top_level_keys(self):
        cfg = load(BASE_CFG)
        assert "model" in cfg, "Missing key 'model'"
        assert "training" in cfg, "Missing key 'training'"

    def test_feature_names_count(self):
        cfg = load(BASE_CFG)
        names = cfg["model"]["feature_names"]
        assert len(names) == 9, f"Expected 9 features, got {len(names)}"

    def test_feature_names_no_duplicates(self):
        cfg = load(BASE_CFG)
        names = cfg["model"]["feature_names"]
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_required_feature_names(self):
        cfg = load(BASE_CFG)
        names = cfg["model"]["feature_names"]
        required = ["body", "upper_wick", "lower_wick", "log_ret_close",
                    "volatility", "max_spread", "mean_obi", "mean_deep_obi", "log_volume"]
        for r in required:
            assert r in names, f"Required feature '{r}' missing from config"

    def test_num_classes(self):
        cfg = load(BASE_CFG)
        assert cfg["model"]["num_classes"] == 3

    def test_class_weights_count(self):
        cfg = load(BASE_CFG)
        weights = cfg["training"]["class_weights"]
        assert len(weights) == 3, f"Expected 3 class weights, got {len(weights)}"

    def test_class_weights_positive(self):
        cfg = load(BASE_CFG)
        for w in cfg["training"]["class_weights"]:
            assert w > 0, f"class_weight {w} is not positive"

    def test_seq_len_valid(self):
        cfg = load(BASE_CFG)
        assert cfg["training"]["seq_len"] in [720, 1440], \
            f"seq_len must be 720 or 1440, got {cfg['training']['seq_len']}"

    def test_gradient_clip_norm_positive(self):
        cfg = load(BASE_CFG)
        clip = cfg["training"]["gradient_clip_norm"]
        assert clip > 0, f"gradient_clip_norm must be positive, got {clip}"

    def test_optimizer_type_valid(self):
        cfg = load(BASE_CFG)
        opt_type = cfg["training"]["optimizer"]["type"]
        assert opt_type in ["Adam", "AdamW", "SGD"], f"Unknown optimizer: {opt_type}"

    def test_scheduler_type_valid(self):
        cfg = load(BASE_CFG)
        sched = cfg["training"]["scheduler"]["type"]
        assert sched in ["CosineAnnealingLR", "StepLR", "OneCycleLR"], \
            f"Unknown scheduler: {sched}"

    def test_early_stopping_patience_positive(self):
        cfg = load(BASE_CFG)
        p = cfg["training"]["early_stopping_patience"]
        assert p >= 1, f"early_stopping_patience must be >= 1, got {p}"


# ── training_config.yaml ──────────────────────────────────────────────────────

class TestTrainingConfig:

    def test_file_exists(self):
        assert TRAIN_CFG.exists(), f"Config not found: {TRAIN_CFG}"

    def test_top_level_keys(self):
        cfg = load(TRAIN_CFG)
        assert "paths" in cfg, "Missing key 'paths'"
        assert "hyperparameters" in cfg, "Missing key 'hyperparameters'"

    def test_required_paths_keys(self):
        cfg = load(TRAIN_CFG)
        for key in ["labelled_dir", "model_output", "scaler_output"]:
            assert key in cfg["paths"], f"Missing paths key: {key}"

    def test_labelled_dir_pattern(self):
        """labelled_dir must reference a labelled_* experiment folder, not raw pre_processed."""
        cfg = load(TRAIN_CFG)
        labelled_dir = cfg["paths"]["labelled_dir"]
        assert "labelled" in labelled_dir, \
            f"labelled_dir should point to a labelled folder: {labelled_dir}"

    def test_model_output_is_pt_file(self):
        cfg = load(TRAIN_CFG)
        assert cfg["paths"]["model_output"].endswith(".pt"), \
            "model_output should be a .pt file"

    def test_scaler_output_is_pkl(self):
        cfg = load(TRAIN_CFG)
        assert cfg["paths"]["scaler_output"].endswith(".pkl"), \
            "scaler_output should be a .pkl file"

    def test_batch_size_safe(self):
        """batch_size > 512 risks OOM with LSTM (validated constraint #4)."""
        cfg = load(TRAIN_CFG)
        bs = cfg["hyperparameters"]["batch_size"]
        assert bs <= 512, f"batch_size={bs} may cause OOM — cap at 512"

    def test_lr_reasonable(self):
        cfg = load(TRAIN_CFG)
        lr = cfg["hyperparameters"]["lr"]
        assert 1e-6 <= lr <= 0.1, f"lr={lr} is outside reasonable range [1e-6, 0.1]"

    def test_dropout_bounded(self):
        cfg = load(TRAIN_CFG)
        drop = cfg["hyperparameters"]["dropout"]
        assert 0.0 <= drop < 1.0, f"dropout={drop} must be in [0, 1)"

    def test_seq_len_positive(self):
        cfg = load(TRAIN_CFG)
        assert cfg["hyperparameters"]["seq_len"] > 0


# ── optimization_config.yaml ──────────────────────────────────────────────────

class TestOptimizationConfig:

    def test_file_exists(self):
        assert OPT_CFG.exists(), f"Config not found: {OPT_CFG}"

    def test_required_keys(self):
        cfg = load(OPT_CFG)
        for key in ["paths", "optimization", "search_space"]:
            assert key in cfg, f"Missing key: {key}"

    def test_n_trials_positive(self):
        cfg = load(OPT_CFG)
        assert cfg["optimization"]["n_trials"] > 0

    def test_metric_is_f1_macro(self):
        """F1 Macro prevents NEUTRAL class from dominating metric."""
        cfg = load(OPT_CFG)
        assert cfg["optimization"]["metric"] == "f1_macro", \
            "Metric must be f1_macro to avoid NEUTRAL class domination"

    def test_batch_size_oom_cap(self):
        """batch_size options must not include 512+ (OOM risk with lstm_hidden=512+seq_len=1440)."""
        cfg = load(OPT_CFG)
        batch_sizes = cfg["search_space"]["batch_size"]
        for bs in batch_sizes:
            assert bs <= 256, f"batch_size={bs} in search space exceeds OOM-safe cap of 256"

    def test_search_space_tcn_channels_valid(self):
        cfg = load(OPT_CFG)
        for ch in cfg["search_space"]["tcn_channels"]:
            assert ch > 0 and ch <= 512

    def test_search_space_lstm_hidden_valid(self):
        cfg = load(OPT_CFG)
        for h in cfg["search_space"]["lstm_hidden"]:
            assert h > 0 and h <= 1024

    def test_study_name_set(self):
        cfg = load(OPT_CFG)
        assert cfg["paths"]["study_name"], "study_name must not be empty"

    def test_db_path_set(self):
        cfg = load(OPT_CFG)
        assert "sqlite" in cfg["paths"]["db_path"], \
            "db_path should be a sqlite:/// URI"


# ── auditor_config.yaml ───────────────────────────────────────────────────────

class TestAuditorConfig:

    def test_file_exists(self):
        assert AUDITOR_CFG.exists(), f"Config not found: {AUDITOR_CFG}"

    def test_required_keys(self):
        cfg = load(AUDITOR_CFG)
        for key in ["paths", "walk_forward", "xgboost"]:
            assert key in cfg, f"Missing key: {key}"

    def test_required_paths_keys(self):
        cfg = load(AUDITOR_CFG)
        for key in ["labelled_dir", "xgb_model_output", "scaler_output", "base_model_checkpoint"]:
            assert key in cfg["paths"], f"Missing paths key: {key}"

    def test_xgb_output_is_json(self):
        cfg = load(AUDITOR_CFG)
        assert cfg["paths"]["xgb_model_output"].endswith(".json")

    def test_n_folds_minimum(self):
        """Walk-forward requires at least 3 folds to be statistically meaningful."""
        cfg = load(AUDITOR_CFG)
        assert cfg["walk_forward"]["n_folds"] >= 3, \
            f"n_folds={cfg['walk_forward']['n_folds']} — minimum 3 for valid walk-forward"

    def test_xgboost_params_positive(self):
        cfg = load(AUDITOR_CFG)
        xgb = cfg["xgboost"]
        assert xgb["n_estimators"] > 0
        assert xgb["max_depth"] > 0
        assert 0 < xgb["learning_rate"] < 1
        assert 0 < xgb["subsample"] <= 1.0
        assert 0 < xgb["colsample_bytree"] <= 1.0

    def test_base_model_checkpoint_is_pt(self):
        cfg = load(AUDITOR_CFG)
        assert cfg["paths"]["base_model_checkpoint"].endswith(".pt")


# ── Cross-config consistency ──────────────────────────────────────────────────

class TestCrossConfigConsistency:

    def test_labelled_dir_matches_training_and_auditor(self):
        """Both training and auditor must use the same labelled experiment."""
        train_cfg   = load(TRAIN_CFG)
        auditor_cfg = load(AUDITOR_CFG)
        train_dir   = train_cfg["paths"]["labelled_dir"]
        auditor_dir = auditor_cfg["paths"]["labelled_dir"]
        assert train_dir == auditor_dir, (
            f"Labelled dir mismatch between training and auditor configs:\n"
            f"  training:  {train_dir}\n"
            f"  auditor:   {auditor_dir}"
        )

    def test_num_features_matches_base_config(self):
        """num_features in base_model_config must equal len(feature_names)."""
        cfg = load(BASE_CFG)
        assert cfg["model"]["num_features"] == len(cfg["model"]["feature_names"])

    def test_class_weights_match_num_classes(self):
        cfg = load(BASE_CFG)
        assert len(cfg["training"]["class_weights"]) == cfg["model"]["num_classes"]
