import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PerformanceJournal:
    """
    Tracks predictions and validates them after the horizon period.
    Saves results to logs/trading_journal.csv for audit.
    """
    def __init__(self, horizon_minutes: int = 15, target_pct: float = 0.003):
        self.horizon_minutes = horizon_minutes
        self.target_pct = target_pct
        self.log_path = Path("logs/trading_journal.csv")
        self.pending_checks: List[Dict] = []
        
        # Initialize file with header if not exists
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(columns=[
                "timestamp_prediction", "symbol", "prediction", "auditor_score", 
                "price_at_prediction", "horizon_deadline", "price_at_deadline",
                "max_price_reached", "min_price_reached", "was_correct", "pnl_observed_pct"
            ])
            df.to_csv(self.log_path, index=False)

    def add_prediction(self, ts: datetime, symbol: str, signal: str, score: float, current_price: float):
        """Registers a new prediction to be checked later."""
        deadline = ts + timedelta(minutes=self.horizon_minutes)
        
        entry = {
            "timestamp_prediction": ts.strftime('%Y-%m-%d %H:%M:%S'),
            "symbol": symbol,
            "prediction": signal,
            "auditor_score": score,
            "price_at_prediction": current_price,
            "horizon_deadline": deadline, # DateTime object for comparison
            "max_price_reached": current_price,
            "min_price_reached": current_price,
            "checked": False
        }
        self.pending_checks.append(entry)
        logger.info(f"📝 Journal: Added prediction {signal} @ {current_price:.2f}. Will check at {deadline.strftime('%H:%M:%S')}")

    def update_market_price(self, current_price: float, current_ts: datetime):
        """Updates high/low for all pending checks and finalize those that reached deadine."""
        ready_to_finalize = []
        
        for entry in self.pending_checks:
            if not entry["checked"]:
                # Update window extremes
                entry["max_price_reached"] = max(entry["max_price_reached"], current_price)
                entry["min_price_reached"] = min(entry["min_price_reached"], current_price)
                
                # Check if deadline reached
                if current_ts >= entry["horizon_deadline"]:
                    ready_to_finalize.append(entry)

        for entry in ready_to_finalize:
            self._finalize_entry(entry, current_price)
            self.pending_checks.remove(entry)

    def _finalize_entry(self, entry: Dict, final_price: float):
        """Calculates success and saves to CSV."""
        pred = entry["prediction"]
        price_init = entry["price_at_prediction"]
        
        # Logic to check if 0.3% goal was met during the window
        # For BUY: max_price / price_init >= 1.003
        # For SELL: min_price / price_init <= 0.997
        
        was_correct = False
        pnl_pct = 0.0
        
        if pred == "BUY":
            pnl_pct = (entry["max_price_reached"] - price_init) / price_init
            was_correct = pnl_pct >= self.target_pct
        elif pred == "SELL":
            pnl_pct = (price_init - entry["min_price_reached"]) / price_init
            was_correct = pnl_pct >= self.target_pct
        else: # NEUTRAL
            was_correct = None # Neutral isn't "right or wrong" in this audit logic
            pnl_pct = (final_price - price_init) / price_init

        # Prepare for CSV
        clean_entry = entry.copy()
        clean_entry["price_at_deadline"] = final_price
        clean_entry["horizon_deadline"] = entry["horizon_deadline"].strftime('%Y-%m-%d %H:%M:%S')
        clean_entry["was_correct"] = was_correct
        clean_entry["pnl_observed_pct"] = pnl_pct * 100
        del clean_entry["checked"]
        
        # Append to CSV
        df = pd.DataFrame([clean_entry])
        df.to_csv(self.log_path, mode='a', header=False, index=False)
        
        from src.cloud.base_model.utils.color_utils import TerminalColors as TC
        status_icon = "✅" if was_correct else "❌"
        res_color = TC.GREEN if was_correct else TC.RED
        
        if was_correct is None: 
            status_icon = "⚪"
            res_color = TC.CYAN
        
        log_msg = f"🔍 Journal CHECK: Pred {pred} @ {entry['timestamp_prediction']} | Result: {status_icon} (Max PnL: {pnl_pct*100:.2f}%)"
        logger.info(TC.color_text(log_msg, res_color))
