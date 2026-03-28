import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import sys
from tqdm import tqdm
import json

# Add project root to sys.path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.execution.inference_service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequentialBacktestEngineV2:
    """
    High-Fidelity Sequential Backtest Engine.
    Simulates trades bar-by-bar to find exact exit points instead of relying on pre-calculated labels.
    """
    def __init__(self, config_path: str):
        project_root = Path(__file__).parents[2]
        master_path = project_root / "src" / "cloud" / "base_model" / "configs" / "master_config.yaml"
        
        with open(master_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        with open(config_path, 'r') as f:
            backtest_cfg = yaml.safe_load(f)
            
        self._merge_configs(self.config, backtest_cfg)
            
        self.inference = InferenceService(self.config)
        self._project_root = Path(self.config.get('project_root', '.'))
        if not self._project_root.is_absolute():
            self._project_root = (project_root / self._project_root).resolve()

    def _merge_configs(self, base, overlay):
        for k, v in overlay.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge_configs(base[k], v)
            else:
                base[k] = v
        
        sim_cfg = self.config.get('simulation', {})
        self.initial_capital = sim_cfg.get('initial_capital', 10000)
        self.fee = sim_cfg.get('trading_fee', 0.0006)
        
        # Strategy Parameters (Volatility multipliers from master_config/training)
        self.pt_mult = sim_cfg.get('pt_multiplier', 0.65)
        self.sl_mult = sim_cfg.get('sl_multiplier', 0.33)
        self.vol_span = 144 # Fixed to match labelling logic (1 day of 10min context)
        self.exit_horizon_ms = 15 * 60 * 1000 # 15 minutes
        
        self.output_dir = Path(self.config['pipeline_paths']['local_data_root']) / "backtest_results_sequential_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_sequential_backtest(self):
        data_dir = Path(self.config['pipeline_paths']['labelled_dir'])
        parquet_files = sorted(list(data_dir.glob("*.parquet")))

        # [VERIFICAÇÃO V2.2] Filtrando apenas o dia 21 sugerido pelo usuário por ser o menor
        parquet_files = [f for f in parquet_files if "2026-03-21" in f.name]
        
        if not parquet_files:
            logger.error(f"❌ No parquet files found in {data_dir}")
            return

        logger.info(f"🚀 [V2] Starting High-Fidelity Backtest on {len(parquet_files)} days...")
        
        balance = self.initial_capital
        equity_curve = []
        all_trades = []
        busy_until = 0 
        
        for pf in tqdm(parquet_files, desc="📅 Simulation Progress"):
            df = pd.read_parquet(pf)
            
            # Feature extraction (same as v1 for consistency)
            base_features = self.config['model']['feature_names']
            context_features = [
                'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
                'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
                'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
            ]
            
            final_context_cols = []
            for f in context_features:
                if f in df.columns: final_context_cols.append(f)
                elif f"alpha_{f}" in df.columns: final_context_cols.append(f"alpha_{f}")
            
            try:
                X_base = df[base_features].values
                X_context = df[final_context_cols].values
                batch_results = self.inference.predict_batch(X_base, X_context)
            except Exception as e:
                logger.error(f"❌ Skipping day {pf.stem} due to error: {e}")
                continue
            
            signals = batch_results['signals'] # 0=SELL, 1=NEUTRAL, 2=BUY
            scores = batch_results['auditor_scores']
            
            # Diagnostics
            directions_idx = np.argmax(batch_results.get('probs_specialist', np.zeros((len(signals), 3))), axis=1)
            n_final_buy = (signals == 2).sum()
            n_final_sell = (signals == 0).sum()
            
            directional_scores = scores[directions_idx != 1]
            max_dir_score = directional_scores.max() if len(directional_scores) > 0 else 0
            logger.info(f"📊 Day {pf.stem} Results: Max Dir Score: {max_dir_score:.4f} | Audited: BUY={n_final_buy}, SELL={n_final_sell}")
            
            timestamps = df['ts'].values
            prices = df['close'].values
            highs = df['high'].values
            lows = df['low'].values

            # Calculation of Volatility EWMA (Parity with labelling process)
            log_ret = np.log(pd.Series(prices)).diff()
            volatility = log_ret.ewm(span=self.vol_span).std().values
            
            S = self.inference.seq_len
            
            # Step through the day
            for i in range(len(signals)):
                idx = i + S - 1
                curr_ts = timestamps[idx]
                
                if curr_ts < busy_until:
                    continue
                
                sig = signals[i]
                if sig == 1: # Neutral
                    continue
                    
                # APPROVED SIGNAL: Simulate Execution with Dynamic Barriers
                entry_price = prices[idx]
                entry_ts = curr_ts
                curr_vol = volatility[idx]
                
                # If vol is NaN (start of day), fallback to a reasonable minimum
                if np.isnan(curr_vol):
                    curr_vol = np.nanmean(volatility[:100]) if not np.all(np.isnan(volatility[:100])) else 0.0003
                
                if sig == 2: # LONG
                    target_tp = entry_price * np.exp(curr_vol * self.pt_mult)
                    target_sl = entry_price * np.exp(-curr_vol * self.sl_mult)
                else: # SHORT
                    target_tp = entry_price * np.exp(-curr_vol * self.pt_mult) # Sell TP is down
                    target_sl = entry_price * np.exp(curr_vol * self.sl_mult)  # Sell SL is up
                
                # --- TRIPLE BARRIER SEARCH (The realism heart) ---
                exit_idx = idx + 1
                outcome = "TIME"
                exit_price = entry_price
                
                while exit_idx < len(prices):
                    curr_bar_ts = timestamps[exit_idx]
                    if (curr_bar_ts - entry_ts) > self.exit_horizon_ms:
                        outcome = "TIME"
                        exit_price = prices[exit_idx]
                        break
                    
                    if sig == 2: # LONG
                        if highs[exit_idx] >= target_tp:
                            outcome = "TP"
                            exit_price = target_tp # Assume limit fill at TP
                            break
                        if lows[exit_idx] <= target_sl:
                            outcome = "SL"
                            exit_price = target_sl # Assume exit at SL
                            break
                    else: # SHORT
                        if lows[exit_idx] <= target_tp:
                            outcome = "TP"
                            exit_price = target_tp
                            break
                        if highs[exit_idx] >= target_sl:
                            outcome = "SL"
                            exit_price = target_sl
                            break
                    exit_idx += 1
                
                # If we reached the end of the day without hitting any barrier
                if exit_idx >= len(prices):
                    exit_idx = len(prices) - 1
                    exit_price = prices[exit_idx]
                    outcome = "DAY_CLOSE"
                
                # --- PNL CALCULATION ---
                if sig == 2: # Long
                    trade_return = (exit_price - entry_price) / entry_price
                else: # Short
                    trade_return = (entry_price - exit_price) / entry_price
                
                pnl_net = trade_return - self.fee
                balance *= (1 + pnl_net)
                busy_until = timestamps[exit_idx] # Only free after exit
                
                all_trades.append({
                    "ts_entry": entry_ts,
                    "ts_exit": timestamps[exit_idx],
                    "type": "BUY" if sig == 2 else "SELL",
                    "price_entry": entry_price,
                    "price_exit": exit_price,
                    "outcome": outcome,
                    "pnl_net": pnl_net,
                    "balance": balance,
                    "duration_min": (timestamps[exit_idx] - entry_ts) / 60000
                })
                
                equity_curve.append({"ts": timestamps[exit_idx], "balance": balance})

        # Save & Final Report
        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            df_trades.to_parquet(self.output_dir.parent / "backtest_results_REALISTIC_V2.parquet")
            self.print_report(df_trades, balance)
        else:
            logger.warning("📭 No trades executed in V2 (Realistic) mode.")

    def print_report(self, df_trades, final_balance):
        total_days = (df_trades['ts_entry'].max() - df_trades['ts_entry'].min()) / (1000 * 60 * 60 * 24)
        wins = df_trades[df_trades['pnl_net'] > 0]
        losses = df_trades[df_trades['pnl_net'] <= 0]
        
        report = {
            "initial_capital": self.initial_capital,
            "final_balance": float(final_balance),
            "roi_pct": float((final_balance / self.initial_capital - 1) * 100),
            "total_trades": len(df_trades),
            "win_rate": len(wins) / len(df_trades) if len(df_trades) > 0 else 0,
            "avg_duration_min": float(df_trades['duration_min'].mean()),
            "total_days": float(total_days)
        }
        
        logger.info("============== [V2] REALISTIC SEQUENTIAL REPORT ==============")
        logger.info(json.dumps(report, indent=4))
        logger.info(f"__ Results saved to data/backtest/L2/backtest_results_REALISTIC_V2.parquet")

if __name__ == "__main__":
    import os
    config_path = os.path.join("src", "backtest", "backtest_config_cloud.yaml")
    engine = SequentialBacktestEngineV2(config_path)
    engine.run_sequential_backtest()
