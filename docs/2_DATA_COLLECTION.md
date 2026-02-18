# 📡 2. Data Collection (Ingestion)

> **Target Audience**: Data Engineers, Pipeline Architects.

This module is responsible for "drinking from the firehose"—ingesting massive amounts of high-frequency market data from Binance Futures.

---

## 📜 Manual do Coletor Histórico (`historical.py`)
Downloads monthly archives of **Aggregated Trades** (`aggTrades`) and **Klines** (1m).

### Execution
```powershell
python -m src.collectors.historical
```

### Core Logic
1.  **Async/Await**: Uses `aiohttp` to download multiple months in parallel (Semaphore=3).
2.  **In-Memory Processing**: Unzips and parses CSVs directly in RAM (`io.BytesIO`) to avoid disk thrashing.
3.  **Skip Existing**: Checks if the target `.parquet` file exists before downloading. Ideally for resuming interrupted jobs.
4.  **404 Handling**:
    > **Note Critical**: Binance Futures (`ETHUSDT`, `BTCUSDT`) only launched around **September 2019**. Requests for dates prior to this (e.g., 2017) will return **404 Not Found**. The script logs this as a warning and proceeds gracefully.

---

## 🌊 Manual do Streamer (`stream.py`)
Connects to Binance WebSocket for real-time data ingestion.

### Execution
```powershell
python -m src.collectors.stream
```

### WebSocket Manager
- **Endpoint**: `wss://fstream.binance.com/ws/{symbol}@depth20@100ms/{symbol}@aggTrade`
- **Multiplexed Stream**: Subscribes to both Order Book Depth (Top 20 levels) and Aggregated Trades simultaneously.

### Real-Time Storage (Buffer & Flush)
To prevent I/O bottlenecks, data is buffered in memory (List of Dicts) and flushed to Parquet in batches.

**Flush Triggers:**
1.  **Time**: Every 60 seconds (`STREAM_FLUSH_INTERVAL_SECONDS`).
2.  **Size**: If buffer exceeds ~20,000 items (prevent OOM).

**Directory Structure:**
```
data/raw/stream/
├── depth/
│   └── 2024-01-01/
│       └── depth_20240101_120000.parquet
└── trades/
    └── 2024-01-01/
        └── trades_20240101_120000.parquet
```

This ensures zero data loss during high volatility while keeping files manageable.
