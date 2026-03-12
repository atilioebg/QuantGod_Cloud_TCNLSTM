import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class VirtualBroker:
    """
    Simulates a brokerage account for Paper Trading.
    - Tracks Virtual Balance (USDT) and Positions.
    - Simulates execution with slippage based on L2 Best Bid/Ask.
    - Calculates real-time P&L.
    """
    def __init__(self, initial_balance: float = 10000.0, fee_pct: float = 0.001):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct # 0.1% Binance Standard Fee
        
        self.position_size = 0.0 # Current amount of Asset (e.g. BTC)
        self.last_position_size = 0.0 # Size of the last closed position
        self.entry_price = 0.0
        self.trades_history: List[Dict] = []
        
        logger.info(f"🏦 VirtualBroker Initialized: ${initial_balance} | Fee: {fee_pct*100}%")

    def execute_signal(self, signal: str, price_snapshot: Dict[str, float], ts: datetime):
        """
        Executes a trade based on the signal and current L2 depth.
        price_snapshot: {'bid': float, 'ask': float}
        """
        if signal == "BUY" and self.position_size == 0:
            self._buy(price_snapshot['ask'], ts) # Buy at Best Ask (taker)
        elif signal == "SELL" and self.position_size > 0:
            self._sell(price_snapshot['bid'], ts) # Sell at Best Bid (taker)
        # NEUTRAL: do nothing or hold existing position

    def _buy(self, price: float, ts: datetime):
        # We use 95% of balance to allow for fees and safety
        available_usdt = self.balance * 0.95
        quantity = available_usdt / price
        fee = available_usdt * self.fee_pct
        
        self.balance -= (available_usdt + fee)
        self.position_size = quantity
        self.entry_price = price
        
        trade = {
            "ts": ts,
            "type": "BUY",
            "price": price,
            "quantity": quantity,
            "fee": fee,
            "balance_after": self.balance
        }
        self.trades_history.append(trade)
        logger.info(f"🟢 [TRADE] BUY {quantity:.6f} @ ${price:.2f} | Fee: ${fee:.2f}")

    def _sell(self, price: float, ts: datetime):
        usdt_value = self.position_size * price
        fee = usdt_value * self.fee_pct
        
        self.balance += (usdt_value - fee)
        pnl = (price - self.entry_price) * self.position_size - (fee + self.trades_history[-1]['fee'])
        
        trade = {
            "ts": ts,
            "type": "SELL",
            "price": price,
            "quantity": self.position_size,
            "fee": fee,
            "pnl": pnl,
            "balance_after": self.balance
        }
        self.trades_history.append(trade)
        logger.info(f"🔴 [TRADE] SELL {self.position_size:.6f} @ ${price:.2f} | P&L: ${pnl:.2f} | Balance: ${self.balance:.2f}")
        
        self.last_position_size = self.position_size
        self.position_size = 0.0
        self.entry_price = 0.0

    def get_stats(self, current_price: float) -> Dict[str, Any]:
        equity = self.balance
        if self.position_size > 0:
            equity += (self.position_size * current_price)
            
        total_pnl = equity - self.initial_balance
        pnl_pct = (total_pnl / self.initial_balance) * 100
        
        return {
            "equity": equity,
            "total_pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "num_trades": len(self.trades_history) // 2,
            "current_balance": self.balance,
            "position": self.position_size,
            "last_position": self.last_position_size
        }
