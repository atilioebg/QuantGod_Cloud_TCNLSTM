"""
test_sequence_dataset.py — Unit tests for the island-aware SequenceDataset.

Validates that:
  1. __len__ is correct for a single continuous island.
  2. __len__ is LESS than (total - seq_len) when there are 2+ islands
     (cross-island sequences must be excluded).
  3. __getitem__ never returns a sequence that crosses an island boundary.
  4. An island smaller than seq_len produces 0 valid sequences and doesn't crash.
  5. Empty dataset (len < seq_len) produces 0 valid sequences.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from src.cloud.base_model.treino.run_specialization import SequenceDataset


SEQ_LEN = 10


def _make_X_y(n: int):
    """Synthetic feature arrays of length n."""
    X = np.random.rand(n, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=n).astype(np.int64)
    return X, y


# ── Test 1: Single island ─────────────────────────────────────────────────────
class TestSingleIsland:

    def test_len_matches_naive_formula(self):
        """With one island only, len == n - seq_len + 1
        (window can start at indices 0..n-seq_len, inclusive)."""
        n = 100
        X, y = _make_X_y(n)
        ids = np.zeros(n, dtype=np.int64)  # all same island
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        expected = n - SEQ_LEN + 1  # 91, not 90
        assert len(ds) == expected, (
            f"Expected {expected}, got {len(ds)}"
        )

    def test_getitem_returns_correct_shape(self):
        n = 100
        X, y = _make_X_y(n)
        ids = np.zeros(n, dtype=np.int64)
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        x_batch, y_label = ds[0]
        assert x_batch.shape == (SEQ_LEN, 5), f"Bad x shape: {x_batch.shape}"
        assert y_label.ndim == 0,             "y_label must be scalar tensor"


# ── Test 2: Two islands ───────────────────────────────────────────────────────
class TestTwoIslands:

    def setup_method(self):
        """Build a dataset with 2 equal-sized islands."""
        self.n = 100
        self.island_size = 50
        X, y = _make_X_y(self.n)
        ids = np.array(
            [0] * self.island_size + [1] * self.island_size, dtype=np.int64
        )
        self.ds = SequenceDataset(X, y, ids, SEQ_LEN)

    def test_len_less_than_naive(self):
        """Cross-island sequences excluded: len < n - seq_len + 1."""
        naive = self.n - SEQ_LEN + 1
        assert len(self.ds) < naive, (
            f"Expected cross-island exclusion: got {len(self.ds)} vs naive {naive}"
        )

    def test_len_equals_two_islands_independently(self):
        """len == 2 * (island_size - seq_len + 1), one per island."""
        expected = 2 * (self.island_size - SEQ_LEN + 1)  # 2 * 41 = 82
        assert len(self.ds) == expected, (
            f"Expected {expected}, got {len(self.ds)}"
        )

    def test_no_cross_island_sequence(self):
        """Every sequence must have the same island_id at start and end."""
        n = 100
        island_size = 50
        X, y = _make_X_y(n)
        ids = np.array(
            [0] * island_size + [1] * island_size, dtype=np.int64
        )
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        for idx in range(len(ds)):
            real = ds.valid_indices[idx]
            start_island = ids[real]
            end_island   = ids[real + SEQ_LEN - 1]
            assert start_island == end_island, (
                f"Sequence at idx={idx} (real={real}) crosses islands: "
                f"{start_island} != {end_island}"
            )


# ── Test 3: Island smaller than seq_len ───────────────────────────────────────
class TestSmallIsland:

    def test_zero_valid_sequences(self):
        """Island of 5 bars with seq_len=10 must produce 0 valid sequences."""
        n = 5
        X, y = _make_X_y(n)
        ids = np.zeros(n, dtype=np.int64)
        ds = SequenceDataset(X, y, ids, SEQ_LEN)  # seq_len=10 > n=5
        assert len(ds) == 0, f"Expected 0, got {len(ds)}"

    def test_indexing_empty_dataset_raises(self):
        n = 5
        X, y = _make_X_y(n)
        ids = np.zeros(n, dtype=np.int64)
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        with pytest.raises(IndexError):
            _ = ds[0]


# ── Test 4: Empty dataset ─────────────────────────────────────────────────────
class TestEmptyDataset:

    def test_zero_len_on_empty(self):
        X, y = _make_X_y(0)
        ids = np.array([], dtype=np.int64)
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        assert len(ds) == 0

    def test_zero_len_on_one_bar(self):
        X, y = _make_X_y(1)
        ids = np.zeros(1, dtype=np.int64)
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        assert len(ds) == 0


# ── Test 5: Getitem correctness ───────────────────────────────────────────────
class TestGetItemCorrectness:

    def test_x_seq_values_match_source(self):
        """x_seq returned must match X[real_idx : real_idx + seq_len]."""
        n = 60
        X, y = _make_X_y(n)
        ids = np.zeros(n, dtype=np.int64)
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        for i in range(min(5, len(ds))):
            x_batch, _ = ds[i]
            real_idx = ds.valid_indices[i]
            expected = torch.from_numpy(X[real_idx: real_idx + SEQ_LEN])
            assert torch.allclose(x_batch, expected), (
                f"Mismatch at idx={i}: got {x_batch[0]}, expected {expected[0]}"
            )

    def test_y_label_is_correct(self):
        """y_label must be y[real_idx + seq_len - 1]."""
        n = 60
        X, y = _make_X_y(n)
        ids = np.zeros(n, dtype=np.int64)
        ds = SequenceDataset(X, y, ids, SEQ_LEN)
        for i in range(min(5, len(ds))):
            _, y_label = ds[i]
            real_idx = ds.valid_indices[i]
            expected = int(y[real_idx + SEQ_LEN - 1])
            assert int(y_label.item()) == expected, (
                f"y_label mismatch at idx={i}: got {y_label.item()}, expected {expected}"
            )
