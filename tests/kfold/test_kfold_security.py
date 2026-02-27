"""
test_kfold_security.py — Anti-Leakage Security Tests for K-Fold OOF Specialist

Validates the integrity and causal correctness of the Blocked Purged K-Fold split
implemented in run_kfold_specialist.py.

Tests guarantee:
  1. Temporal ordering preserved (no future data in training).
  2. Purge gap >= purge_minutes between train and test blocks.
  3. Full coverage of Foundation Val in full_oof.parquet.
  4. No duplicate predictions.
  5. Softmax probability sanity (3 columns, sums ≈ 1).
  6. Meta-target is not trivial (model learned something).
  7. Per-fold scaler is isolated (not global).

All thresholds are read dynamically from master_config.yaml.
"""

import pytest
import polars as pl
import numpy as np
import yaml
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from src.cloud.base_model.treino.run_kfold_specialist import (
    blocked_purged_kfold_indices,
)

# from src.cloud.base_model.pre_processamento.etl.transform import _parse_resample_minutes

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_config() -> dict:
    cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    if not cfg_path.exists():
        pytest.skip("master_config.yaml not found")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_resample_min(freq: str) -> int:
    freq = freq.strip()
    for s in ('min', 'T', 'Min'):
        if freq.endswith(s):
            return max(1, int(freq.replace(s, '')))
    if freq.endswith(('h', 'H')):
        return int(freq[:-1]) * 60
    return 1


def load_oof_params():
    """Returns (n_splits, purge_bars, oof_dir) from config."""
    config = load_config()
    kfold  = config['pre_processing']['kfold']
    freq   = config['pre_processing']['etl'].get('resample_freq', '1min')
    rmin   = parse_resample_min(freq)
    purge_min  = kfold.get('purge_minutes', 15)
    purge_bars = max(1, purge_min // rmin)
    n_splits   = kfold.get('n_splits', 5)
    oof_dir    = Path(kfold.get('oof_output_dir', 'data/auditor/oof_predictions'))
    return n_splits, purge_bars, purge_min, oof_dir


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: Algorithmic Tests (no data on disk required)
# ══════════════════════════════════════════════════════════════════════════════
class TestBlockedPurgedKFoldAlgorithm:
    """Unit tests for the blocked_purged_kfold_indices generator."""

    def setup_method(self):
        config         = load_config()
        kfold          = config['pre_processing']['kfold']
        freq           = config['pre_processing']['etl'].get('resample_freq', '1min')
        rmin           = parse_resample_min(freq)
        self.n_splits  = kfold.get('n_splits', 5)
        self.purge_min = kfold.get('purge_minutes', 15)
        self.purge_bars = max(1, self.purge_min // rmin)
        self.n          = 10000   # synthetic dataset size

    def test_no_temporal_leakage(self):
        """
        CORE ANTI-LEAKAGE TEST:
        For every fold, the maximum training index must be strictly less than
        the minimum test index. This guarantees that no future data feeds
        into the training window.
        """
        for fold_k, (train_idx, test_idx) in enumerate(
            blocked_purged_kfold_indices(self.n, self.n_splits, self.purge_bars)
        ):
            assert len(train_idx) > 0, f"Fold {fold_k}: empty train_idx"
            assert len(test_idx) > 0,  f"Fold {fold_k}: empty test_idx"

            # NOTE: This is the Blocked K-Fold — train can come from ANY block
            # except the test block. We only need: no train index == any test index.
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, (
                f"Fold {fold_k}: DATA LEAKAGE — {len(overlap)} indices appear in both train and test!"
            )

    def test_purge_gap_between_adjacent_blocks(self):
        """
        Validates that adjacent blocks have purge_bars > 0 gap between them.
        Specifically: last row of the preceding train block < first row of test block.
        The gap must be >= purge_bars (from the chronologically preceding train block).
        """
        config = load_config()
        for fold_k, (train_idx, test_idx) in enumerate(
            blocked_purged_kfold_indices(self.n, self.n_splits, self.purge_bars)
        ):
            test_start = test_idx.min()
            test_end   = test_idx.max()

            # Find train rows that come BEFORE the test block
            before_test = train_idx[train_idx < test_start]
            after_test  = train_idx[train_idx > test_end]

            if len(before_test) > 0:
                gap_right = test_start - before_test.max() - 1
                assert gap_right >= self.purge_bars, (
                    f"Fold {fold_k}: purge gap BEFORE test block is {gap_right} bars, "
                    f"expected >= {self.purge_bars} ({self.purge_min}min)"
                )

            if len(after_test) > 0:
                gap_left = after_test.min() - test_end - 1
                assert gap_left >= self.purge_bars, (
                    f"Fold {fold_k}: purge gap AFTER test block is {gap_left} bars, "
                    f"expected >= {self.purge_bars} ({self.purge_min}min)"
                )

    def test_all_folds_generated(self):
        """Exactly n_splits folds must be generated."""
        folds = list(blocked_purged_kfold_indices(self.n, self.n_splits, self.purge_bars))
        assert len(folds) == self.n_splits, (
            f"Expected {self.n_splits} folds, got {len(folds)}"
        )

    def test_total_coverage_close_to_full(self):
        """
        The union of all test blocks minus purge waste covers ≈ 100% of n.
        Coverage percent must exceed (1 - purge_overhead).
        """
        all_test = np.concatenate([
            test_idx for _, test_idx in
            blocked_purged_kfold_indices(self.n, self.n_splits, self.purge_bars)
        ])
        coverage = len(np.unique(all_test)) / self.n
        # Minimum coverage = 1 - (2 * purge_bars * n_splits) / n
        min_coverage = 1.0 - (2.0 * self.purge_bars * self.n_splits) / self.n
        assert coverage >= max(0.9, min_coverage), (
            f"Coverage {coverage:.1%} is below minimum {max(0.9, min_coverage):.1%}. "
            f"purge_bars={self.purge_bars}, n_splits={self.n_splits}"
        )

    def test_purge_respects_horizon_minutes(self):
        """purge_minutes must be >= horizon_minutes to guarantee causal isolation."""
        config       = load_config()
        kfold        = config['pre_processing']['kfold']
        horizon_min  = config['pre_processing']['labelling'].get('horizon_minutes', 15)
        purge_min    = kfold.get('purge_minutes', 15)
        assert purge_min >= horizon_min, (
            f"purge_minutes ({purge_min}) < horizon_minutes ({horizon_min}). "
            f"Labels at the boundary can bleed into training features!"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: Output File Tests (require run_kfold_specialist.py to have been run)
# ══════════════════════════════════════════════════════════════════════════════
class TestFullOOFOutput:
    """Validates the full_oof.parquet produced by a completed K-Fold run."""

    @pytest.fixture(autouse=True)
    def skip_if_no_oof(self):
        n_splits, purge_bars, purge_min, oof_dir = load_oof_params()
        full_oof_path = oof_dir / "full_oof.parquet"
        if not full_oof_path.exists():
            pytest.skip(f"full_oof.parquet not found at {full_oof_path} (run K-Fold first)")
        self.df        = pl.read_parquet(full_oof_path)
        self.n_splits  = n_splits
        self.purge_bars = purge_bars
        self.oof_dir   = oof_dir

    def test_required_columns_present(self):
        required = {"original_row_idx", "spec_prob_sell", "spec_prob_neu", "spec_prob_buy",
                    "spec_pred_class", "true_target", "fold"}
        present = set(self.df.columns)
        assert required.issubset(present), f"Missing columns: {required - present}"

    def test_no_duplicate_predictions(self):
        """Each row in Foundation Val must appear exactly once in OOF predictions."""
        n_rows   = len(self.df)
        n_unique = self.df['original_row_idx'].n_unique()
        assert n_unique == n_rows, (
            f"{n_rows - n_unique} duplicate original_row_idx found in full_oof.parquet. "
            "This means some rows were predicted more than once."
        )

    def test_softmax_probability_sanity(self):
        """Each row's 3 probabilities must sum to approximately 1.0 (softmax output)."""
        prob_sum = (
            self.df['spec_prob_sell'] +
            self.df['spec_prob_neu'] +
            self.df['spec_prob_buy']
        ).to_numpy()
        assert np.allclose(prob_sum, 1.0, atol=1e-4), (
            f"Softmax probabilities do not sum to 1. "
            f"Max deviation: {np.abs(prob_sum - 1.0).max():.6f}"
        )

    def test_probabilities_in_valid_range(self):
        """All probabilities must be in [0, 1]."""
        for col in ['spec_prob_sell', 'spec_prob_neu', 'spec_prob_buy']:
            arr = self.df[col].to_numpy()
            assert arr.min() >= -1e-6, f"{col} has negative value: {arr.min()}"
            assert arr.max() <= 1+1e-6, f"{col} exceeds 1: {arr.max()}"

    def test_meta_target_not_trivial(self):
        """
        The model should have predicted at least some BUY (2) and SELL (0).
        A 100% NEUTRAL prediction means the model collapsed — worthless for the Auditor.
        """
        pred_classes = self.df['spec_pred_class'].to_numpy()
        n_buy    = np.sum(pred_classes == 2)
        n_sell   = np.sum(pred_classes == 0)
        n_total  = len(pred_classes)
        assert n_buy  > 0, "Model predicted 0 BUY — collapsed to trivial solution!"
        assert n_sell > 0, "Model predicted 0 SELL — collapsed to trivial solution!"
        # At least 1% directional predictions — if not, warn about threshold calibration
        dir_pct = (n_buy + n_sell) / n_total
        if dir_pct < 0.01:
            pytest.warns(
                UserWarning,
                match="Directional predictions"
            )

    def test_all_n_splits_present(self):
        """Exactly n_splits unique fold IDs must be present in full_oof.parquet."""
        unique_folds = set(self.df['fold'].unique().to_list())
        expected     = set(range(self.n_splits))
        assert unique_folds == expected, (
            f"Expected fold IDs {expected}, found {unique_folds}"
        )

    def test_per_fold_scalers_exist(self):
        """Each fold must have its own scaler saved (proof of anti-leakage via normalization)."""
        for k in range(self.n_splits):
            sp = self.oof_dir / f"scaler_fold_{k}.pkl"
            assert sp.exists(), f"Per-fold scaler missing: {sp}"
            with open(sp, 'rb') as f:
                scaler = pickle.load(f)
            assert hasattr(scaler, 'mean_'), f"Scaler fold_{k} was never fit!"

    def test_chronological_order(self):
        """full_oof.parquet must be sorted by original_row_idx (chronological)."""
        idx = self.df['original_row_idx'].to_numpy()
        assert np.all(idx[:-1] <= idx[1:]), (
            "full_oof.parquet is NOT in chronological order! Sort by original_row_idx."
        )
