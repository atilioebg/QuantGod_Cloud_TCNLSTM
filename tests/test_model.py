"""
test_model.py — Unit tests for Hybrid_TCN_LSTM (base model).

Validates WITHOUT torch (pure numpy where possible) and WITH torch when needed.
Tests are structured to run fast even locally — no GPU, no large data loading.

Covered:
  - Model import and instantiation
  - Forward pass output shape (logits + probs)
  - Probability simplex invariant (probs sum to 1)
  - Causal convolution — output length must match input length (no future leakage)
  - Gradient flow (logits have grad on training input)
  - Determinism in eval mode (same input → same output)
  - Multi-config instantiation (various tcn_channels / lstm_hidden combos)
  - Parameter count sanity (must be > 0 trainable params)
"""

import pytest

# Skip entire module if torch is not installed (local dev without GPU env)
torch = pytest.importorskip("torch", reason="torch not installed — skipping model tests")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cloud.base_model.models.model import CausalConv1d, TCNBlock, Hybrid_TCN_LSTM

SEQ_LEN     = 720
NUM_FEATURES = 9
NUM_CLASSES  = 3
BATCH_SIZE   = 4


@pytest.fixture(scope="module")
def model_default():
    """Default Hybrid_TCN_LSTM instance in eval mode."""
    m = Hybrid_TCN_LSTM(num_features=NUM_FEATURES, seq_len=SEQ_LEN)
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_input():
    """Synthetic batch tensor (B, T, F) = (4, 720, 9)."""
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)


# ── Import & Instantiation ───────────────────────────────────────────────────

class TestModelInstantiation:

    def test_default_instantiation(self):
        m = Hybrid_TCN_LSTM()
        assert m is not None

    def test_custom_channels(self):
        for tcn_ch in [32, 64, 128]:
            m = Hybrid_TCN_LSTM(tcn_channels=tcn_ch)
            assert m is not None

    def test_custom_lstm_hidden(self):
        for hidden in [128, 256, 512]:
            m = Hybrid_TCN_LSTM(lstm_hidden=hidden)
            assert m is not None

    def test_trainable_parameters_exist(self, model_default):
        trainable = sum(p.numel() for p in model_default.parameters() if p.requires_grad)
        assert trainable > 0, "Model has no trainable parameters"

    def test_parameter_count_reasonable(self, model_default):
        """Model should be between 100K and 20M params for the default config."""
        count = sum(p.numel() for p in model_default.parameters() if p.requires_grad)
        assert 100_000 < count < 20_000_000, f"Unexpected param count: {count:,}"


# ── Forward Pass — Output Shape ──────────────────────────────────────────────

class TestForwardPassShape:

    def test_output_is_dict(self, model_default, dummy_input):
        with torch.no_grad():
            out = model_default(dummy_input)
        assert isinstance(out, dict), "Output must be a dict"

    def test_output_keys(self, model_default, dummy_input):
        with torch.no_grad():
            out = model_default(dummy_input)
        assert "logits" in out, "Missing key 'logits'"
        assert "probs"  in out, "Missing key 'probs'"

    def test_logits_shape(self, model_default, dummy_input):
        with torch.no_grad():
            out = model_default(dummy_input)
        assert out["logits"].shape == (BATCH_SIZE, NUM_CLASSES), \
            f"Bad logits shape: {out['logits'].shape}, expected ({BATCH_SIZE}, {NUM_CLASSES})"

    def test_probs_shape(self, model_default, dummy_input):
        with torch.no_grad():
            out = model_default(dummy_input)
        assert out["probs"].shape == (BATCH_SIZE, NUM_CLASSES), \
            f"Bad probs shape: {out['probs'].shape}, expected ({BATCH_SIZE}, {NUM_CLASSES})"

    def test_batch_size_1(self, model_default):
        """Edge case: single-sample inference (live inference scenario)."""
        x = torch.randn(1, SEQ_LEN, NUM_FEATURES)
        with torch.no_grad():
            out = model_default(x)
        assert out["logits"].shape == (1, NUM_CLASSES)
        assert out["probs"].shape  == (1, NUM_CLASSES)

    def test_seq_len_1440(self):
        """Model must accept longer sequences (Optuna trials with seq_len=1440)."""
        m = Hybrid_TCN_LSTM(seq_len=1440)
        m.eval()
        x = torch.randn(2, 1440, NUM_FEATURES)
        with torch.no_grad():
            out = m(x)
        assert out["logits"].shape == (2, NUM_CLASSES)


# ── Probability Simplex Invariants ───────────────────────────────────────────

class TestProbabilityInvariants:

    def test_probs_sum_to_one(self, model_default, dummy_input):
        with torch.no_grad():
            out = model_default(dummy_input)
        sums = out["probs"].sum(dim=-1)   # (B,)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5), \
            f"Probs do not sum to 1.0 — max deviation: {(sums - 1).abs().max():.2e}"

    def test_probs_non_negative(self, model_default, dummy_input):
        with torch.no_grad():
            out = model_default(dummy_input)
        assert (out["probs"] >= 0).all(), "Found negative probability values"

    def test_probs_bounded(self, model_default, dummy_input):
        with torch.no_grad():
            out = model_default(dummy_input)
        assert (out["probs"] <= 1.0 + 1e-6).all(), "Found probability > 1.0"


# ── Causal Convolution — No Future Leakage ───────────────────────────────────

class TestCausalConv:

    def test_causal_output_length_matches_input(self):
        """CausalConv1d must not change the temporal dimension."""
        for dilation in [1, 2, 4, 8]:
            conv = CausalConv1d(in_channels=9, out_channels=64, kernel_size=3, dilation=dilation)
            x = torch.randn(2, 9, 720)   # (B, C, T)
            out = conv(x)
            assert out.shape[-1] == 720, \
                f"dilation={dilation}: output T={out.shape[-1]}, expected 720"

    def test_causal_independence_future(self):
        """
        Verify causality: perturbing inputs at t > k should NOT change output at t = k.
        Strategy: run two forward passes — one clean, one with noise added after position 50.
        Output[:, k] must be identical for k < 50.
        """
        torch.manual_seed(0)
        conv = CausalConv1d(in_channels=9, out_channels=9, kernel_size=3, dilation=1)
        conv.eval()

        x1 = torch.randn(1, 9, 720)
        x2 = x1.clone()
        # Add noise AFTER position 50 only in x2
        x2[:, :, 51:] += 10.0

        with torch.no_grad():
            out1 = conv(x1)
            out2 = conv(x2)

        # Outputs at positions 0..48 must be identical (kernel=3, dilation=1 → receptive=3)
        assert torch.allclose(out1[:, :, :49], out2[:, :, :49], atol=1e-5), \
            "CausalConv1d: output at early timesteps changed when only future inputs were perturbed"


# ── Determinism in Eval Mode ─────────────────────────────────────────────────

class TestDeterminism:

    def test_eval_mode_deterministic(self, model_default, dummy_input):
        """Same input must produce identical output in eval mode (no Dropout stochasticity)."""
        with torch.no_grad():
            out1 = model_default(dummy_input)
            out2 = model_default(dummy_input)
        assert torch.allclose(out1["logits"], out2["logits"]), \
            "Model in eval mode produced different logits on identical input"
        assert torch.allclose(out1["probs"], out2["probs"]), \
            "Model in eval mode produced different probs on identical input"


# ── Gradient Flow ────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_logits_have_gradient(self):
        """Verify the computational graph is connected end-to-end (training scenario)."""
        model = Hybrid_TCN_LSTM()
        model.train()
        x = torch.randn(2, SEQ_LEN, NUM_FEATURES, requires_grad=False)
        out = model(x)
        loss = out["logits"].sum()
        loss.backward()
        # Every model parameter must have a gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for param: {name}"


# ── FocalLossWithSmoothing ───────────────────────────────────────────────────

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cloud.base_model.treino.losses import FocalLossWithSmoothing, compute_alpha_from_labels


class TestFocalLoss:

    def test_instantiation_no_alpha(self):
        """FocalLoss must instantiate without alpha (equal weighting)."""
        loss_fn = FocalLossWithSmoothing(alpha=None, gamma=2.0, smoothing=0.1)
        assert loss_fn is not None

    def test_instantiation_with_alpha(self):
        """FocalLoss must instantiate with an alpha tensor."""
        alpha = torch.tensor([3.03, 0.43, 3.03])
        loss_fn = FocalLossWithSmoothing(alpha=alpha, gamma=2.0, smoothing=0.1)
        assert loss_fn is not None

    def test_forward_returns_scalar(self):
        """Forward pass with reduction='mean' must return a scalar tensor."""
        loss_fn = FocalLossWithSmoothing(alpha=None, gamma=2.0, smoothing=0.1)
        logits  = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
        assert loss.item() > 0.0, "Loss should be positive"

    def test_loss_lower_when_confident_and_correct(self):
        """
        When the model is very confident AND correct (logit >> for correct class),
        focal loss should be lower than when the model is uncertain.
        """
        loss_fn = FocalLossWithSmoothing(alpha=None, gamma=2.0, smoothing=0.0)
        targets = torch.tensor([0, 1, 2])

        # Confident-correct: high logit for the right class
        logits_good = torch.tensor([
            [10.0, -5.0, -5.0],  # class 0 correct with high confidence
            [-5.0, 10.0, -5.0],
            [-5.0, -5.0, 10.0],
        ])
        # Uncertain: near-zero logits
        logits_uncertain = torch.zeros(3, 3)

        loss_good      = loss_fn(logits_good, targets)
        loss_uncertain = loss_fn(logits_uncertain, targets)

        assert loss_good.item() < loss_uncertain.item(), (
            f"Expected confident-correct loss ({loss_good.item():.4f}) < "
            f"uncertain loss ({loss_uncertain.item():.4f})"
        )

    def test_reduction_none_returns_per_sample(self):
        """reduction='none' must return a tensor with one value per sample."""
        loss_fn = FocalLossWithSmoothing(alpha=None, gamma=2.0, smoothing=0.1, reduction='none')
        logits  = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (8,), f"Expected shape (8,), got {loss.shape}"

    def test_gradient_flows_through_focal_loss(self):
        """Backward pass through FocalLoss must propagate gradients to model params."""
        model = Hybrid_TCN_LSTM()
        model.train()
        alpha    = torch.tensor([3.03, 0.43, 3.03])
        loss_fn  = FocalLossWithSmoothing(alpha=alpha, gamma=2.0, smoothing=0.1)
        x        = torch.randn(2, SEQ_LEN, NUM_FEATURES)
        targets  = torch.randint(0, NUM_CLASSES, (2,))
        out      = model(x)
        loss     = loss_fn(out["logits"], targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient through FocalLoss for: {name}"


class TestComputeAlpha:

    def test_output_shape(self):
        """compute_alpha_from_labels must return a tensor of shape (num_classes,)."""
        y = np.array([0, 1, 2, 1, 1, 0, 2, 1])
        alpha = compute_alpha_from_labels(y, num_classes=3)
        assert alpha.shape == (3,), f"Expected shape (3,), got {alpha.shape}"

    def test_output_dtype(self):
        """Alpha must be float32."""
        y = np.array([0, 1, 2, 1, 1])
        alpha = compute_alpha_from_labels(y, num_classes=3)
        assert alpha.dtype == torch.float32

    def test_minority_class_has_higher_weight(self):
        """
        With 80% Neutral and 10% each for SELL/BUY, their weights
        must be higher than Neutral's weight (inverse frequency).
        """
        # Simulate 100 samples: 80 neutral, 10 sell, 10 buy
        y = np.array([0] * 10 + [1] * 80 + [2] * 10)
        alpha = compute_alpha_from_labels(y, num_classes=3)
        assert alpha[0] > alpha[1], "SELL weight should exceed NEUTRAL weight"
        assert alpha[2] > alpha[1], "BUY weight should exceed NEUTRAL weight"

    def test_balanced_data_equal_weights(self):
        """When classes are perfectly balanced, all weights must be equal (≈ 1.0)."""
        y = np.array([0, 1, 2] * 100)
        alpha = compute_alpha_from_labels(y, num_classes=3)
        expected = 300 / (3 * 100)  # = 1.0
        assert torch.allclose(alpha, torch.tensor([expected, expected, expected], dtype=torch.float32), atol=1e-4)

    def test_formula_correctness(self):
        """Verify exact formula: alpha_i = total / (num_classes * count_i)."""
        y = np.array([0] * 10 + [1] * 80 + [2] * 10)  # total = 100
        alpha = compute_alpha_from_labels(y, num_classes=3)
        expected_sell    = 100 / (3 * 10)   # ~3.333
        expected_neutral = 100 / (3 * 80)   # ~0.417
        assert abs(alpha[0].item() - expected_sell)    < 1e-4
        assert abs(alpha[1].item() - expected_neutral) < 1e-4
        assert abs(alpha[2].item() - expected_sell)    < 1e-4
