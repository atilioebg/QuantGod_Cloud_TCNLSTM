
"""
feature_engineering_meta.py — XGBoost Auditor Meta-Feature Engineering

Engineering Constraint #2: NO WARM-UP in live inference.
All indicators (RSI, EMA distances, Bollinger %B, ATR normalized) are computed
directly from the micro_price series already present in the 720-step tensor that
the base model holds in memory. This eliminates any need to wait for warm-up
periods (e.g., 50 minutes for EMA50) during live trading.

Usage:
    from src.cloud.auditor_model.feature_engineering_meta import extract_meta_features
    features = extract_meta_features(micro_price_series, base_model_output)
"""

import numpy as np
import torch
from typing import Dict, Optional


# ─── Pure-numpy indicator functions ──────────────────────────────────────────
# All operate on 1D arrays of shape (T,) derived from the 720-step tensor.
# These are deterministic and require no external state or warm-up time.

def _rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    RSI computed over the last `period` price changes from the full series.
    Uses Wilder's smoothed method approximated over available history.
    Guaranteed valid with min T >= period+1 (always true with T=720).
    """
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(prices: np.ndarray, period: int) -> float:
    """
    EMA of the last timestamp computed iteratively from the full series.
    Works on T=720, where T >> period. Uses standard multiplier k=2/(period+1).
    """
    k = 2.0 / (period + 1)
    ema_val = prices[0]
    for p in prices[1:]:
        ema_val = p * k + ema_val * (1 - k)
    return float(ema_val)


def _ema_distance_pct(prices: np.ndarray, period: int) -> float:
    """% distance of current close from its EMA: (close - EMA) / EMA."""
    ema_val = _ema(prices, period)
    close   = prices[-1]
    return float((close - ema_val) / (ema_val + 1e-9))


def _bollinger_pct_b(prices: np.ndarray, period: int = 20, n_std: float = 2.0) -> float:
    """
    Bollinger %B = (price - lower) / (upper - lower)
    Values: 0=at lower band, 1=at upper band, >1 or <0 = outside bands.
    Computed over last `period` candles from the 720-step tensor.
    """
    window = prices[-period:]
    mean   = np.mean(window)
    std    = np.std(window)
    upper  = mean + n_std * std
    lower  = mean - n_std * std
    band_range = upper - lower
    if band_range < 1e-9:
        return 0.5  # Flat window, neutral position
    return float((prices[-1] - lower) / band_range)


def _atr_normalized(prices: np.ndarray, period: int = 14) -> float:
    """
    Normalized ATR = ATR(14) / close_price.
    True Range for 1D micro_price series (no separate high/low):
    TR[t] = |close[t] - close[t-1]|  (simplified, valid for resampled micro-price).
    Normalized by current close to make it scale-invariant.
    MACD excluded per Constraint #2 (multicolinearity with EMAs).
    """
    tr = np.abs(np.diff(prices[-period - 1:]))
    atr = np.mean(tr[-period:])
    return float(atr / (prices[-1] + 1e-9))


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy of model output probabilities. High = uncertain."""
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# ─── Main extraction function ─────────────────────────────────────────────────

def extract_meta_features(
    micro_price_series: np.ndarray,
    base_model_probs: np.ndarray,
    last_step_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract 14 meta-features for the XGBoost auditor from the 720-step micro_price tensor.

    Args:
        micro_price_series: np.ndarray of shape (720,) — micro_price values from the
                            current sequence, extracted from the model input tensor at t=0..719.
        base_model_probs:   np.ndarray of shape (3,) — [prob_sell, prob_neutral, prob_buy]
                            from Hybrid_TCN_LSTM.forward()["probs"].
        last_step_features: np.ndarray of shape (9,) — normalized feature values at t=720
                            (last timestep of the sequence). If None, last 7 features are 0.

    Returns:
        features: np.ndarray of shape (14,)

    Feature index map:
        [0]  prob_sell
        [1]  prob_neutral
        [2]  prob_buy
        [3]  entropy
        [4]  body            (from last_step_features[0])
        [5]  upper_wick      (from last_step_features[1])
        [6]  lower_wick      (from last_step_features[2])
        [7]  log_ret_close   (from last_step_features[3])
        [8]  volatility      (from last_step_features[4])
        [9]  mean_obi        (from last_step_features[6])
        [10] mean_deep_obi   (from last_step_features[7])
        [11] rsi_14          (computed from micro_price_series)
        [12] ema_9_dist      (% distance from EMA9)
        [13] ema_50_dist     (% distance from EMA50)
    Note: max_spread[5] and log_volume[8] excluded (see implementation_plan.md).

    Indicators use the full 720-step history — zero warm-up needed in live inference.
    """
    assert micro_price_series.ndim == 1 and len(micro_price_series) >= 50, \
        f"micro_price_series must be 1D with length >= 50, got {micro_price_series.shape}"
    assert len(base_model_probs) == 3, "base_model_probs must have shape (3,)"

    prices = micro_price_series.astype(np.float64)

    # Group 1: Base model probabilities
    prob_sell    = float(base_model_probs[0])
    prob_neutral = float(base_model_probs[1])
    prob_buy     = float(base_model_probs[2])

    # Group 2: Model uncertainty
    entropy = _entropy(base_model_probs)

    # Group 3: Last-step original features (7 of 9, excluding max_spread and log_volume)
    if last_step_features is not None:
        # FEATURE_NAMES = [body, upper_wick, lower_wick, log_ret_close, volatility,
        #                  max_spread, mean_obi, mean_deep_obi, log_volume]
        # Indices:           0        1          2          3          4         5        6       7          8
        body         = float(last_step_features[0])
        upper_wick   = float(last_step_features[1])
        lower_wick   = float(last_step_features[2])
        log_ret_close= float(last_step_features[3])
        volatility   = float(last_step_features[4])
        mean_obi     = float(last_step_features[6])
        mean_deep_obi= float(last_step_features[7])
    else:
        body = upper_wick = lower_wick = log_ret_close = volatility = mean_obi = mean_deep_obi = 0.0

    # Group 4: Momentum/trend/volatility indicators from micro_price tensor (NO WARM-UP)
    rsi_14    = _rsi(prices, period=14)
    ema_9_dist = _ema_distance_pct(prices, period=9)
    ema_50_dist= _ema_distance_pct(prices, period=50)
    # Note: Bollinger %B and ATR computed here but NOT included in final 14 features
    # per implementation plan. Available for future expansion:
    # bollinger_pct_b = _bollinger_pct_b(prices)
    # atr_norm = _atr_normalized(prices)

    return np.array([
        prob_sell,
        prob_neutral,
        prob_buy,
        entropy,
        body,
        upper_wick,
        lower_wick,
        log_ret_close,
        volatility,
        mean_obi,
        mean_deep_obi,
        rsi_14,
        ema_9_dist,
        ema_50_dist,
    ], dtype=np.float32)


# Feature names in order (for XGBoost feature_names param and logging)
META_FEATURE_NAMES = [
    "prob_sell",
    "prob_neutral",
    "prob_buy",
    "entropy",
    "body",
    "upper_wick",
    "lower_wick",
    "log_ret_close",
    "volatility",
    "mean_obi",
    "mean_deep_obi",
    "rsi_14",
    "ema_9_dist",
    "ema_50_dist",
]


def extract_meta_features_batch(
    X_sequences: np.ndarray,
    base_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Batch extraction of meta-features for XGBoost training.

    Args:
        X_sequences:  np.ndarray of shape (N, 720, 9) — normalized sequences
        base_model:   Hybrid_TCN_LSTM (already trained, eval mode)
        device:       torch.device
        batch_size:   inference batch size

    Returns:
        meta_features: np.ndarray of shape (N, 14)

    The micro_price series is not stored in the 9 normalized features directly.
    It's reconstructed from the stationarized `log_ret_close` column (index 3)
    by cumulatively exponentiating from an arbitrary base (we use base=1.0).
    This gives relative micro_price changes, sufficient for RSI, EMA distances, etc.
    """
    base_model.eval()
    all_meta = []

    with torch.no_grad():
        for i in range(0, len(X_sequences), batch_size):
            batch = torch.tensor(X_sequences[i: i + batch_size], dtype=torch.float32).to(device)
            with torch.amp.autocast(device.type):
                out = base_model(batch)
            probs_np = out["probs"].cpu().numpy()   # (B, 3)

            # Reconstruct relative micro_price from log_ret_close (col index 3)
            log_rets = X_sequences[i: i + batch_size, :, 3]  # (B, 720)
            # Cumulative exp from arbitrary base=1.0 → relative price series
            micro_prices = np.exp(np.cumsum(log_rets, axis=1))  # (B, 720)

            for j in range(len(probs_np)):
                last_step = X_sequences[i + j, -1, :]  # (9,) — normalized last step
                meta = extract_meta_features(micro_prices[j], probs_np[j], last_step)
                all_meta.append(meta)

    return np.array(all_meta, dtype=np.float32)  # (N, 14)
