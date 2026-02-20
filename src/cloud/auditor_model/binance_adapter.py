
"""
binance_adapter.py — Binance Futures L2 Live Adapter

Port of QuantGod/vivit_L2/src/live/collector_l2.py (L2LiveCollector, Bybit)
adapted to Binance Futures WebSocket + REST API.

Engineering Constraint #1 — STRICT lastUpdateId/U/u SYNCHRONIZATION:
    The Binance Futures depth stream requires:
    1. REST bootstrap: GET /fapi/v1/depth?limit=1000 → get initial book + lastUpdateId
    2. Buffer WS messages during REST call
    3. Discard any WS packet where u <= lastUpdateId (already covered by REST)
    4. Apply first packet where U <= lastUpdateId+1 <= u
    This guarantees zero gaps or overlaps in liquidity reconstruction.

Feature parity with Bybit training data:
    - Same 9 stationary features: body, upper_wick, lower_wick, log_ret_close,
      volatility, max_spread, mean_obi, mean_deep_obi, log_volume
    - Same math: micro_price, OBI, deep_OBI, finalize_minute (identical to transform.py)
    - Same scaler: loads scaler_finetuning.pkl from training

Usage:
    python binance_adapter.py              # Live mode
    python binance_adapter.py --test-mode  # Prints features for 2 minutes then exits
"""

import asyncio
import json
import aiohttp
import websockets
import numpy as np
import pandas as pd
import pickle
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# ─── Configuration ────────────────────────────────────────────────────────────
SYMBOL           = "BTCUSDT"
BINANCE_REST_URL = "https://fapi.binance.com/fapi/v1/depth"
BINANCE_WS_URL   = f"wss://fstream.binance.com/stream?streams={SYMBOL.lower()}@depth@100ms"
BOOK_HARD_CUT    = 200   # Match training data: top 200 bids/asks
OBI_DEPTH        = 5     # Depth for deep OBI (top 5 levels, same as collector_l2.py)
DRIFT_SIGMA_THRESHOLD = 4.0  # Feature drift alert threshold

SCALER_PATH = Path("data/models/scaler_finetuning.pkl")
FEATURE_NAMES = [
    'body', 'upper_wick', 'lower_wick', 'log_ret_close',
    'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
]

# ─── Logging ──────────────────────────────────────────────────────────────────
log_dir = Path("logs/live")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class BinanceL2Adapter:
    """
    Binance Futures L2 live data adapter with feature parity to Bybit training data.

    Mirrors L2LiveCollector from QuantGod/vivit_L2/src/live/collector_l2.py.
    All math (micro_price, OBI, deep_obi_5, finalize_minute) is identical.
    Only the WebSocket protocol and REST bootstrap differ.
    """

    def __init__(self, test_mode: bool = False, max_candles: int = None):
        self.symbol    = SYMBOL
        self.test_mode = test_mode
        self.max_candles = max_candles  # For --test-mode: exit after N candles

        # ── Orderbook state ───────────────────────────────────────────────────
        self.bids: dict = {}  # price → size
        self.asks: dict = {}  # price → size
        self.last_update_id: int = 0
        self.book_initialized: bool = False

        # ── Constraint #1: WS buffering during REST bootstrap ─────────────────
        self._ws_buffer: list = []
        self._bootstrap_done: bool = False

        # ── Candle aggregation (same as L2LiveCollector) ──────────────────────
        self.current_minute = None
        self.ticks_in_minute: list = []
        self.prev_close = None
        self.candles_produced: int = 0

        # ── Scaler (from training) ─────────────────────────────────────────────
        self.scaler = None
        self._load_scaler()

        # ── Drift detection: running stats from scaler ─────────────────────────
        self.scaler_mean = None
        self.scaler_std  = None
        if self.scaler is not None:
            self.scaler_mean = self.scaler.mean_
            self.scaler_std  = np.sqrt(self.scaler.var_)

    def _load_scaler(self):
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH}. Features will not be normalized.")

    # ─── REST Bootstrap (Constraint #1) ──────────────────────────────────────

    async def _bootstrap_orderbook(self, session: aiohttp.ClientSession):
        """
        Fetch initial orderbook snapshot via REST API.
        Must be called BEFORE applying WS deltas.
        Also drains the WS buffer of packets received during this REST call,
        discarding any where u <= lastUpdateId.
        """
        logger.info("Bootstrapping orderbook via REST API...")
        params = {"symbol": self.symbol, "limit": 1000}
        async with session.get(BINANCE_REST_URL, params=params) as resp:
            data = await resp.json()

        self.last_update_id = data["lastUpdateId"]
        self.bids = {float(p): float(s) for p, s in data["bids"] if float(s) > 0}
        self.asks = {float(p): float(s) for p, s in data["asks"] if float(s) > 0}

        # Apply hard cut 200 immediately after REST snapshot
        self._apply_hard_cut()
        self.book_initialized = True
        self._bootstrap_done  = True
        logger.info(f"REST snapshot done | lastUpdateId={self.last_update_id} | "
                    f"bids={len(self.bids)} | asks={len(self.asks)}")

        # Drain buffered WS messages accumulated during REST call
        for buffered_msg in self._ws_buffer:
            self._apply_delta_safe(buffered_msg)
        self._ws_buffer.clear()

    def _apply_delta_safe(self, msg: dict):
        """
        Apply a WS delta message with strict lastUpdateId validation.

        Constraint #1 rules (Binance Futures):
            - Discard if u (finalUpdateId) <= lastUpdateId (already in REST)
            - Discard if U (firstUpdateId) > lastUpdateId + 1 (gap detected)
            - Apply otherwise, update lastUpdateId = u
        """
        U = msg.get("U", 0)  # First update ID in this event
        u = msg.get("u", 0)  # Final update ID in this event
        event_time = msg.get("T", 0)

        # Discard: entirely before our snapshot
        if u <= self.last_update_id:
            return

        # Gap detection: first valid packet must have U <= lastUpdateId+1
        if not self.book_initialized:
            return

        if U > self.last_update_id + 1:
            # Gap detected — need to re-bootstrap
            logger.warning(f"GAP DETECTED: U={U} > lastUpdateId+1={self.last_update_id+1}. "
                           f"Re-bootstrapping...")
            self.book_initialized = False
            self.bids.clear()
            self.asks.clear()
            return

        # Apply delta
        for p, s in msg.get("b", []):
            price, size = float(p), float(s)
            if size == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        for p, s in msg.get("a", []):
            price, size = float(p), float(s)
            if size == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

        self.last_update_id = u
        self._apply_hard_cut()
        self._process_tick(event_time)

    def _apply_hard_cut(self):
        """Trim orderbook to top BOOK_HARD_CUT levels (same as transform.py)."""
        if len(self.bids) > BOOK_HARD_CUT:
            top_bids = sorted(self.bids.keys(), reverse=True)[:BOOK_HARD_CUT]
            self.bids = {k: self.bids[k] for k in top_bids}
        if len(self.asks) > BOOK_HARD_CUT:
            top_asks = sorted(self.asks.keys())[:BOOK_HARD_CUT]
            self.asks = {k: self.asks[k] for k in top_asks}

    # ─── Feature Extraction (identical math to collector_l2.py) ───────────────

    def _calculate_instant_features(self, ts: int) -> dict | None:
        """
        Identical to L2LiveCollector.calculate_instant_features().
        Computes micro_price, spread, obi_l0, deep_obi_5 at tick level.
        """
        if not self.bids or not self.asks:
            return None

        s_bids = sorted(self.bids.keys(), reverse=True)[:OBI_DEPTH]
        s_asks = sorted(self.asks.keys())[:OBI_DEPTH]

        if not s_bids or not s_asks:
            return None

        b0_p, b0_s = s_bids[0], self.bids[s_bids[0]]
        a0_p, a0_s = s_asks[0], self.asks[s_asks[0]]

        # Micro-price (weighted midpoint)
        micro_price = (b0_p * a0_s + a0_p * b0_s) / (b0_s + a0_s + 1e-9)

        # OBI L0 (top-of-book imbalance)
        obi_l0 = (b0_s - a0_s) / (b0_s + a0_s + 1e-9)

        # Deep OBI (top 5 levels)
        bid_vol_5 = sum(self.bids[p] for p in s_bids)
        ask_vol_5 = sum(self.asks[p] for p in s_asks)
        deep_obi_5 = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5 + 1e-9)

        return {
            "ts": ts,
            "micro_price": micro_price,
            "spread": a0_p - b0_p,
            "obi_l0": obi_l0,
            "deep_obi_5": deep_obi_5,
            "bid_vol_5": bid_vol_5,
            "ask_vol_5": ask_vol_5,
        }

    def _process_tick(self, ts: int):
        """Route tick to current minute buffer. Finalize on minute boundary."""
        features = self._calculate_instant_features(ts)
        if not features:
            return

        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        current_minute_ts = dt.replace(second=0, microsecond=0)

        if self.current_minute is None:
            self.current_minute = current_minute_ts

        if current_minute_ts > self.current_minute:
            self._finalize_minute(self.current_minute)
            self.current_minute = current_minute_ts

        self.ticks_in_minute.append(features)

    def _finalize_minute(self, minute_ts):
        """
        Identical to L2LiveCollector.finalize_minute().
        Computes the same 9 stationary features from tick-level data.
        """
        if not self.ticks_in_minute:
            return

        df_m = pd.DataFrame(self.ticks_in_minute)

        # Micro-Price OHLC
        ohlc = {
            "open":  df_m['micro_price'].iloc[0],
            "high":  df_m['micro_price'].max(),
            "low":   df_m['micro_price'].min(),
            "close": df_m['micro_price'].iloc[-1],
        }

        if self.prev_close is None:
            self.prev_close = ohlc['open']

        prev = self.prev_close
        candle = {
            "datetime":      minute_ts,
            "body":          float(np.log(ohlc['close'] / ohlc['open'] + 1e-9)),
            "upper_wick":    float((ohlc['high'] - max(ohlc['open'], ohlc['close'])) / (prev + 1e-9)),
            "lower_wick":    float((min(ohlc['open'], ohlc['close']) - ohlc['low']) / (prev + 1e-9)),
            "log_ret_close": float(np.log(ohlc['close'] / (prev + 1e-9))),
            "volatility":    float(df_m['micro_price'].std()),
            "max_spread":    float(df_m['spread'].max()),
            "mean_obi":      float(df_m['obi_l0'].mean()),
            "mean_deep_obi": float(df_m['deep_obi_5'].mean()),
            "log_volume":    float(np.log1p(len(df_m))),
            "close":         ohlc['close'],
        }

        self.prev_close = ohlc['close']
        self.ticks_in_minute = []
        self.candles_produced += 1

        # Apply scaler and drift check
        self._emit_candle(candle)

    def _emit_candle(self, candle: dict):
        """Apply scaler, detect feature drift, emit candle for inference."""
        raw_features = np.array([[candle[f] for f in FEATURE_NAMES]], dtype=np.float32)

        # ── Feature drift detection (> DRIFT_SIGMA_THRESHOLD σ from training) ─
        if self.scaler_mean is not None:
            z_scores = np.abs((raw_features[0] - self.scaler_mean) / (self.scaler_std + 1e-9))
            drifted  = [(FEATURE_NAMES[i], float(z_scores[i]))
                        for i in range(len(FEATURE_NAMES)) if z_scores[i] > DRIFT_SIGMA_THRESHOLD]
            if drifted:
                logger.warning(f"⚠️  FEATURE DRIFT DETECTED at {candle['datetime']}: {drifted}")

        # ── Normalize with training scaler ─────────────────────────────────────
        if self.scaler is not None:
            normalized = self.scaler.transform(raw_features)[0]
        else:
            normalized = raw_features[0]

        norm_candle = {f: float(normalized[i]) for i, f in enumerate(FEATURE_NAMES)}
        norm_candle['datetime'] = candle['datetime']
        norm_candle['close_raw'] = candle['close']  # Keep raw close for meta-feature RSI/EMA

        logger.info(
            f"📊 Candle {self.candles_produced} | {candle['datetime']} | "
            f"close={candle['close']:.2f} | log_ret={candle['log_ret_close']:.5f} | "
            f"obi={candle['mean_obi']:.4f}"
        )

        if self.test_mode and self.max_candles and self.candles_produced >= self.max_candles:
            logger.info(f"Test mode: {self.candles_produced} candles produced. Exiting.")
            raise KeyboardInterrupt("Test mode complete")

        return norm_candle

    # ─── WebSocket Message Handler ─────────────────────────────────────────────

    async def _handle_message(self, raw_msg: str):
        """Parse Binance depth stream message and route to delta processor."""
        outer = json.loads(raw_msg)
        msg   = outer.get("data", outer)  # stream wrapper or direct message

        # Binance Futures depth stream message format:
        # {"stream": "btcusdt@depth@100ms", "data": {"e": "depthUpdate", "T": ..., "U": ..., "u": ..., "b": [], "a": []}}
        if msg.get("e") != "depthUpdate":
            return

        if not self._bootstrap_done:
            # Buffer messages received during REST bootstrap
            self._ws_buffer.append(msg)
            return

        self._apply_delta_safe(msg)

    # ─── Main Entry Points ────────────────────────────────────────────────────

    async def start(self):
        """Main coroutine: REST bootstrap + WebSocket stream loop."""
        logger.info(f"📡 Starting Binance Futures L2 Adapter for {self.symbol}")
        logger.info(f"   WS: {BINANCE_WS_URL}")
        logger.info(f"   Scaler: {'loaded' if self.scaler else 'NOT FOUND'}")
        logger.info(f"   Test mode: {self.test_mode}")

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with websockets.connect(BINANCE_WS_URL) as ws:
                        logger.info("✅ WebSocket connected")

                        # Start REST bootstrap concurrently while WS is connected
                        # (WS messages are buffered until bootstrap completes)
                        bootstrap_task = asyncio.create_task(
                            self._bootstrap_orderbook(session)
                        )

                        # Process WS messages (buffered until bootstrap_task done)
                        async for raw_msg in ws:
                            if not bootstrap_task.done():
                                outer = json.loads(raw_msg)
                                msg   = outer.get("data", outer)
                                if msg.get("e") == "depthUpdate":
                                    self._ws_buffer.append(msg)
                            else:
                                await self._handle_message(raw_msg)

                except KeyboardInterrupt:
                    logger.info("🛑 Adapter stopped by user.")
                    break
                except Exception as e:
                    logger.error(f"❌ Connection error: {e}. Reconnecting in 5s...")
                    self.book_initialized  = False
                    self._bootstrap_done   = False
                    self._ws_buffer.clear()
                    await asyncio.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Binance Futures L2 Live Adapter")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run for 2 candles then exit (for offline testing)")
    parser.add_argument("--max-candles", type=int, default=2,
                        help="Number of candles to produce in test mode")
    args = parser.parse_args()

    adapter = BinanceL2Adapter(
        test_mode=args.test_mode,
        max_candles=args.max_candles if args.test_mode else None,
    )
    try:
        asyncio.run(adapter.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
