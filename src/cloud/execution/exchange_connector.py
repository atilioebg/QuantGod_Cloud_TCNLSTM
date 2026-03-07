import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import websockets
from collections import deque

logger = logging.getLogger(__name__)

class DataBuffer:
    """
    Thread-safe/Async-safe buffer for real-time exchange data.
    Maintains the latest L2 snapshot and a rolling window of recent trades.
    """
    def __init__(self, max_trades: int = 5000):
        self.latest_depth: Dict[str, Any] = {}
        self.trades: deque = deque(maxlen=max_trades)
        self.last_update_ts: float = 0
        self._lock = asyncio.Lock()

    async def update_depth(self, data: Dict[str, Any]):
        async with self._lock:
            self.latest_depth = data
            self.last_update_ts = time.time()

    async def add_trade(self, data: Dict[str, Any]):
        async with self._lock:
            self.trades.append(data)
            self.last_update_ts = time.time()

    async def get_snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "depth": self.latest_depth,
                "trades": list(self.trades),
                "timestamp": self.last_update_ts
            }

class ExchangeConnector:
    """
    Connects to Binance WebSocket streams (L2 Depth and Trades).
    Handles automatic reconnection and data steering to the DataBuffer.
    """
    def __init__(self, symbol: str, buffer: DataBuffer):
        self.symbol = symbol.lower()
        self.buffer = buffer
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    async def _handle_depth(self, data: Dict[str, Any]):
        # data: {'lastUpdateId': ..., 'bids': [[price, qty], ...], 'asks': ...}
        await self.buffer.update_depth(data)

    async def _handle_trade(self, data: Dict[str, Any]):
        # data: {'e': 'trade', 'E': ..., 's': 'BTCUSDT', 't': ..., 'p': '...', 'q': '...', ...}
        await self.buffer.add_trade(data)

    async def _listen(self, stream_url: str):
        while self.is_running:
            try:
                async with websockets.connect(stream_url) as ws:
                    logger.info(f"🔌 Connected to stream: {stream_url}")
                    while self.is_running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # Route based on stream type or content
                        if "depth" in stream_url:
                            await self._handle_depth(data)
                        elif "trade" in stream_url:
                            await self._handle_trade(data)
                            
            except Exception as e:
                logger.error(f"❌ WebSocket Error ({stream_url}): {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def start(self):
        """Starts the multiple stream tasks."""
        self.is_running = True
        # Combined stream URL for efficiency: depth20@100ms and trade
        combined_url = f"wss://stream.binance.com:9443/stream?streams={self.symbol}@depth20@100ms/{self.symbol}@trade"
        
        async def _combined_listener():
            while self.is_running:
                try:
                    async with websockets.connect(combined_url) as ws:
                        logger.info(f"🔌 Connected to combined stream: {combined_url}")
                        while self.is_running:
                            msg = await ws.recv()
                            payload = json.loads(msg)
                            stream_name = payload.get('stream', '')
                            data = payload.get('data', {})
                            
                            if "depth" in stream_name:
                                await self._handle_depth(data)
                            elif "trade" in stream_name:
                                await self._handle_trade(data)
                except Exception as e:
                    logger.error(f"❌ Combined WebSocket Error: {e}. Reconnecting in 5s...")
                    await asyncio.sleep(5)

        self._task = asyncio.create_task(_combined_listener())
        logger.info(f"🚀 ExchangeConnector started for {self.symbol.upper()}")

    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
        logger.info(f"🛑 ExchangeConnector stopped.")

# Quick self-test if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        buf = DataBuffer()
        conn = ExchangeConnector("btcustdt", buf)
        conn.start()
        
        try:
            for _ in range(10):
                await asyncio.sleep(2)
                snapshot = await buf.get_snapshot()
                d_len = len(snapshot['depth'].get('bids', []))
                t_len = len(snapshot['trades'])
                print(f"DEBUG: Snapshot at {datetime.now()} | Depth Bids: {d_len} | Total Trades: {t_len}")
        except KeyboardInterrupt:
            pass
        finally:
            conn.stop()

    asyncio.run(main())
