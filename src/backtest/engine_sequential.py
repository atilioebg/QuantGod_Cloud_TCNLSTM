import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import json

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
            base_features = self.inference.arch_params['feature_columns']
            # Simplification: we expect context features to be handled or empty if not in model
            context_features = [c for c in df.columns if c.startswith('alpha_')]
            
            # Predict whole day at once (GPU optimization)
            X_base = df[base_features].values
            X_context = df[context_features].values
            batch_results = self.inference.predict_batch(X_base, X_context)
            
            signals = batch_results['signals'] # 0=SELL, 1=NEUTRAL, 2=BUY
            scores = batch_results['auditor_scores']
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
