# ⚙️ 3. Data Engineering (ETL)

> **Target Audience**: Quants, Data Scientists.

This document details the transformation of raw market data into the high-dimensional tensors that feed the QuantGod brain.

---

## 🦅 Anatomia do Tensor 6D
Our model sees the market as a **6-Channel Video**.
**Shape**: `(Batch, Time=96, Channels=6, Height=128)`

### The 6 Channels
| Channel | Name | Source | Description | Normalization |
|:---|:---|:---|:---|:---|
| **0** | **Bids** | Order Book | Log-Liquidity on the Bid side. | `Log1p(vol) / 10.0` |
| **1** | **Asks** | Order Book | Log-Liquidity on the Ask side. | `Log1p(vol) / 10.0` |
| **2** | **OFI Raw** | Order Flow | Raw Order Flow Imbalance. | `Tanh(val / 10.0)` |
| **3** | **Price Raw** | Market | Relative Price Deviation from Snap Mean. | `(P - Avg) / (Avg + 1e-9) * 100` |
| **4** | **OFI Wavelet** | **Denoised** | OFI cleaned via `db4` Wavelet. | `Tanh(val / 10.0)` |
| **5** | **Price Wavelet** | **Denoised** | Price cleaned via `db4` Wavelet. | `(P_clean - Avg) * 100` |

### Wavelet Denoising Math
We apply **Soft Thresholding** using the Daubechies 4 (`db4`) wavelet.
1.  **Decompose**: Break signal into approximation and detail coefficients.
2.  **Threshold**: Calculate $\sigma = \text{median}(|details|) / 0.6745$.
3.  **Shrink**: Set small coefficients to zero based on $\sigma \sqrt{2 \ln(N)}$.
4.  **Reconstruct**: Build the clean signal.

---

## 🧠 Meta-Features (XGBoost Input)
While the Tensor feeds the Deep Learning model, we generate a parallel tabular dataset for the Meta-Model (XGBoost).
**Path**: `data/processed/meta/meta_{ts}.parquet`

### Indicators Dictionary
| Feature | Type | Logic | Rationale |
|:---|:---|:---|:---|
| **RSI_14** | Momentum | Relative Strength Index (14). | Detect overbought/oversold conditions. |
| **ADX_14** | Trend | Average Directional Index (14). | Measures trend strength (regardless of direction). |
| **EMA_9/21/50/200** | Trend | Exponential Moving Averages. | Multi-timeframe trend alignment. |
| **dist_ema200** | Trend | `(Close - EMA200) / EMA200`. | Measuring extension from the long-term mean. |
| **ATR_14** | Volatility | Average True Range. | Adapts stop-loss and profit targets to volatility. |
| **BB_Width** | Volatility | Bollinger Band Width. | Squeeze detection (low volatility precedes explosion). |
| **OFI_SMA_10** | Microstructure | Simple Moving Average of Order Flow. | Sustained buying/selling pressure. |

---

## 🔗 Integridade Referencial (The Triad)
We enforce a strict 1:1 mapping between datasets:
1.  **Tensor** (`.pt` or `numpy`): The video input.
2.  **Meta-Features** (`.parquet`): The tabular context.
3.  **Target** (`.csv`): The ground truth label.

> **Critical Fault Tolerance**: If any component is missing for a timestamp `T`, the entire sample is discarded during training to prevent "Garbage In, Garbage Out".
