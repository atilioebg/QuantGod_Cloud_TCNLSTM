# 🛠️ 1. Setup & Environment

> **Target Audience**: DevOps, MLOps, System Engineers.

## 🖥️ Hardware Requirements
To run QuantGod in production (Training & Inference), the following specs are recommended:

- **GPU**: NVIDIA RTX 3060 (12GB) or higher.
    - *Reason*: 6D Tensors with 96 time-steps consume significant VRAM during training batches. 12GB+ is comfortable for Batch Size 32-64.
- **RAM**: 32GB DDR4+.
    - *Reason*: `pandas_ta` and Polars operations on high-frequency data (millions of rows) are memory-intensive.
- **Storage**: NVMe SSD (1TB+).
    - *Reason*: Fast I/O is critical for loading Parquet shards during training.

---

## 🛡️ Instalação Blindada (The Ironclad Setup)

### 1. Create Virtual Environment
Isolate dependencies to avoid conflicts.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. The "Dance of PyTorch" (CUDA 12.x)
**CRITICAL**: Do not just run `pip install torch`. You must explicitly target the CUDA version.
```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*Verify installation:*
```python
python -c "import torch; print(torch.cuda.is_available())"
# Should return: True
```

### 3. Install Critical Libraries
QuantGod relies on specific libraries for signal processing and tensor manipulation.
```powershell
pip install -r requirements.txt
```
*Key Dependencies:*
- `PyWavelets`: For `db4` signal denoising.
- `pandas_ta`: High-performance technical analysis.
- `polars`: Blazing fast ETL engine.
- `binance-connector`: Official API wrapper.

---

## 🔐 Environment Configuration (.env)
Create a `.env` file in the project root. This file controls the system's "nervous system".

| Variable | Description | Default |
|:---|:---|:---|
| **ENV** | `development` or `production`. Controls logging verbosity. | `development` |
| **SYMBOL** | Trading pair (e.g., `BTCUSDT`). | `BTCUSDT` |
| **WS_URL** | Binance Futures WebSocket URL. | `wss://fstream.binance.com/ws` |
| **HISTORICAL_START_DATE** | Start date for backfill. Note: BTC Futures started Sep 2019. | `2017-08-01` |
| **BINANCE_API_KEY** | (Optional) For signed requests. | `...` |
| **BINANCE_API_SECRET** | (Optional) For signed requests. | `...` |

**Template:**
```ini
ENV=development
SYMBOL=BTCUSDT
WS_URL=wss://fstream.binance.com/ws
HISTORICAL_START_DATE=2019-09-01
```
