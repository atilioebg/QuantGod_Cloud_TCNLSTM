import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import sys
from tqdm import tqdm
from datetime import datetime, timedelta
import json

# Add project root to sys.path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.execution.inference_service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequentialBacktestEngine:
    def __init__(self, config_path: str):
        # Merge master_config with the provided backtest config
        project_root = Path(__file__).parents[2]
        master_path = project_root / "src" / "cloud" / "base_model" / "configs" / "master_config.yaml"
        
        with open(master_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        with open(config_path, 'r') as f:
            backtest_cfg = yaml.safe_load(f)
            
        # Recursive merge of backtest_cfg into self.config
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
        
        # Simulation Parameters
        sim_cfg = self.config.get('simulation', {})
        self.initial_capital = sim_cfg.get('initial_capital', 10000)
        self.fee = sim_cfg.get('trading_fee', 0.0006) # Default 6 bps if not specified
        
        # Strategy Parameters (Matching Triple Barrier used in ETL)
        self.tp_ret = 0.0020  # 20 bps
        self.sl_ret = 0.0010  # 10 bps
        self.exit_horizon_ms = 15 * 60 * 1000 # 15 minutes in ms
        
        self.output_dir = Path(self.config['pipeline_paths']['local_data_root']) / "backtest_results_sequential"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_sequential_backtest(self):
        data_dir = Path(self.config['pipeline_paths']['labelled_dir'])
        parquet_files = sorted(list(data_dir.glob("*.parquet")))
        
        if not parquet_files:
            logger.error(f"❌ No parquet files found in {data_dir}")
            return

        logger.info(f"🚀 Starting Sequential Backtest on {len(parquet_files)} days...")
        
        balance = self.initial_capital
        equity_curve = []
        all_trades = []
        busy_until = 0 # timestamp ms
        
        for pf in tqdm(parquet_files, desc="📅 Processing Days"):
            df = pd.read_parquet(pf)
            
            # Use columns expected by engine
            base_features = self.config['model']['feature_names']
            
            # Auditor context features (14 technical indicators)
            context_features = [
                'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
                'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
                'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
            ]
            
            # Try to find columns (they might or might not have 'alpha_' prefix)
            final_context_cols = []
            for f in context_features:
                if f in df.columns:
                    final_context_cols.append(f)
                elif f"alpha_{f}" in df.columns:
                    final_context_cols.append(f"alpha_{f}")
                else:
                    # If not found, we might have a problem, but let's see what we have
                    pass
            
            if len(final_context_cols) != 14:
                logger.warning(f"⚠️ expected 14 context features, found {len(final_context_cols)}")
                # FALLBACK: if we can't find them, use startswith('alpha_') just in case
                if not final_context_cols:
                    final_context_cols = [c for c in df.columns if c.startswith('alpha_')]
            
            try:
                X_base = df[base_features].values
                X_context = df[final_context_cols].values
                
                if X_context.shape[1] != 14:
                     logger.error(f"❌ Dimension mismatch: Found {X_context.shape[1]} context features, need 14.")
                     logger.error(f"Available columns: {df.columns.tolist()[:30]}...")
                     return

                batch_results = self.inference.predict_batch(X_base, X_context)
            except Exception as e:
                logger.error(f"❌ Error during inference prep: {e}")
                logger.error(f"Available columns: {df.columns.tolist()[:30]}...")
                return
            
            signals = batch_results['signals'] # 0=SELL, 1=NEUTRAL, 2=BUY
            scores = batch_results['auditor_scores']
            
            # Diagnostic Info
            directions_idx = np.argmax(batch_results.get('probs_specialist', np.zeros((len(signals), 3))), axis=1)
            n_raw_buy = (directions_idx == 2).sum()
            n_raw_sell = (directions_idx == 0).sum()
            n_raw_neu = (directions_idx == 1).sum()
            
            n_final_buy = (signals == 2).sum()
            n_final_sell = (signals == 0).sum()
            max_score = scores.max() if len(scores) > 0 else 0
            
            # Novo: Log de score máximo apenas para sinais direcionais (BUY ou SELL)
            directional_scores = scores[directions_idx != 1]
            max_dir_score = directional_scores.max() if len(directional_scores) > 0 else 0
            
            logger.info(f"📊 Day {pf.stem} Stats:")
            logger.info(f"   - Max Auditor Score (Global): {max_score:.4f}")
            logger.info(f"   - Max Auditor Score (Directional): {max_dir_score:.4f}")
            logger.info(f"   - Raw Specialist: BUY={n_raw_buy}, SELL={n_raw_sell}, NEUTRAL={n_raw_neu}")
            logger.info(f"   - Final (Audited): BUY={n_final_buy}, SELL={n_final_sell}")
            
            timestamps = df['ts'].values
            prices = df['close'].values
            targets = df['target'].values
            
            S = self.inference.seq_len
            
            for i in range(len(signals)):
                idx = i + S - 1
                curr_ts = timestamps[idx]
                
                # State Machine Logic
                if curr_ts < busy_until:
                    continue
                
                # Check for signal
                sig = signals[i]
                score = scores[i]
                
                if sig == 1: # Neutral
                    continue
                    
                # Trading Decision
                if sig == 2: # BUY
                    # Outcome from Triple Barrier (Target 2 = TP, 0 = SL, 1 = Time)
                    target = targets[idx]
                    if target == 2:
                        pnl_raw = self.tp_ret
                        duration = self.exit_horizon_ms / 2 # Mean estimated time to hit TP
                    elif target == 0:
                        pnl_raw = -self.sl_ret
                        duration = self.exit_horizon_ms / 2
                    else:
                        pnl_raw = 0.0
                        duration = self.exit_horizon_ms
                    
                    pnl_net = pnl_raw - self.fee
                    balance *= (1 + pnl_net)
                    busy_until = curr_ts + duration
                    
                    all_trades.append({
                        "ts_entry": curr_ts,
                        "type": "BUY",
                        "price_entry": prices[idx],
                        "target": target,
                        "pnl_net": pnl_net,
                        "balance": balance
                    })
                    
                elif sig == 0: # SELL
                    # Outcome (Target 0 = TP for Sell, 2 = SL for Sell)
                    target = targets[idx]
                    if target == 0:
                        pnl_raw = self.tp_ret
                        duration = self.exit_horizon_ms / 2
                    elif target == 2:
                        pnl_raw = -self.sl_ret
                        duration = self.exit_horizon_ms / 2
                    else:
                        pnl_raw = 0.0
                        duration = self.exit_horizon_ms
                        
                    pnl_net = pnl_raw - self.fee
                    balance *= (1 + pnl_net)
                    busy_until = curr_ts + duration
                    
                    all_trades.append({
                        "ts_entry": curr_ts,
                        "type": "SELL",
                        "price_entry": prices[idx],
                        "target": target,
                        "pnl_net": pnl_net,
                        "balance": balance
                    })

                equity_curve.append({"ts": curr_ts, "balance": balance})

        # Save Results
        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            df_trades.to_parquet(self.output_dir.parent / "backtest_results_SEQUENTIAL_COMPLETE.parquet")
            
            self.print_report(df_trades, balance)
        else:
            logger.warning("📭 No trades executed in sequential mode.")

    def print_report(self, df_trades, final_balance):
        total_days = (df_trades['ts_entry'].max() - df_trades['ts_entry'].min()) / (1000 * 60 * 60 * 24)
        wins = df_trades[df_trades['pnl_net'] > 0]
        losses = df_trades[df_trades['pnl_net'] <= 0]
        
        report = {
            "initial_capital": self.initial_capital,
            "final_balance": float(final_balance),
            "total_profit": float(final_balance - self.initial_capital),
            "roi_pct": float((final_balance / self.initial_capital - 1) * 100),
            "total_trades": len(df_trades),
            "win_rate": len(wins) / len(df_trades) if len(df_trades) > 0 else 0,
            "profit_factor": abs(wins['pnl_net'].sum() / losses['pnl_net'].sum()) if len(losses) > 0 else float('inf'),
            "avg_trades_per_day": len(df_trades) / total_days if total_days > 0 else 0
        }
        
        logger.info("============== SEQUENTIAL BACKTEST REPORT ==============")
        logger.info(json.dumps(report, indent=4))
        logger.info(f"__ Results saved to data/backtest/L2/backtest_results_SEQUENTIAL_COMPLETE.parquet")

if __name__ == "__main__":
    import os
    config_path = os.path.join("src", "backtest", "backtest_config_cloud.yaml")
    engine = SequentialBacktestEngine(config_path)
    engine.run_sequential_backtest()
