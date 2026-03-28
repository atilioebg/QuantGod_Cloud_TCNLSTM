import torch
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import gc
from numpy.lib.stride_tricks import sliding_window_view

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parents[2].absolute()
sys.path.append(str(PROJECT_ROOT))

# Reuse the production inference service logic
from src.cloud.execution.inference_service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        # We wrap the production inference service
        # This ensures the backtest uses the EXACT SAME logic as live
        self.inference = InferenceService(config)
        self.feature_names = config['model']['feature_names']
        self.auditor_features = config['model']['auditor_features']
        
        self.results = []
        self.equity_curve = []
        
    def run_backtest_file(self, file_path):
        """
        Runs the backtest on a single labelled parquet file.
        """
        df = pl.read_parquet(file_path).to_pandas()
        logger.info(f"📊 Processing {file_path.name} ({len(df)} bars)")
        
        # ── Setup ──
        # We need to extract the 30 base features and the auditor meta-features
        base_features = self.feature_names
        # The auditor context starts after the 6 logits (3 base + 3 spec)
        context_features = self.auditor_features[6:] 
        
        # ── Iterative Simulation ──
        # We simulate the decision bar by bar to respect the Sequential nature
        # (Though we could batch the neural layers, let's keep it simple and accurate first)
        
        # Prepare data for faster access
        X_base = df[base_features].values
        X_context = df[context_features].values
        Y_target = df['target'].values
        timestamps = df['ts'].values
        prices = df['close'].values
        
        # ── Batch Prediction (The Speed Boost) ──
        logger.info(f"🧠 Running fully optimized batch inference for {len(X_base)} bars...")
        
        # Instant windowing and scaling logic moved to InferenceService for cleanliness
        # Just send the RAW full sequence
        batch_results = self.inference.predict_batch(X_base, X_context)
        
        # ── Log Results (CPU Simulation Loop) ──
        day_results = []
        signals = batch_results['signals']
        scores = batch_results['auditor_scores']
        probs = batch_results['probs_specialist']
        
        directions_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}
        S = self.inference.seq_len
        
        for i in tqdm(range(len(signals)), desc="📈 Trade Simulation", leave=False):
            idx = i + S - 1 # Original index in the dataframe
            day_results.append({
                "ts": timestamps[idx],
                "price": prices[idx],
                "target": Y_target[idx],
                "signal": directions_map[signals[i]],
                "probs_spec_buy": float(probs[i][2]),
                "probs_spec_sell": float(probs[i][0]),
                "auditor_score": float(scores[i])
            })
            
        return day_results

    def analyze_performance(self, all_results):
        df_res = pd.DataFrame(all_results)
        if df_res.empty:
            return "No signals generated."
            
        # Calculate PnL (Simplified Triple Barrier Outcome)
        # target 0: Sell hit first, target 2: Buy hit first, target 1: Neutral/Stale
        
        def calculate_pnl(row):
            if row['signal'] == "BUY":
                if row['target'] == 2: return 1.0 # Hit TP
                if row['target'] == 0: return -1.0 # Hit SL (assuming 1:1 or as per config)
                return 0.0 # Stale
            if row['signal'] == "SELL":
                if row['target'] == 0: return 1.0 # Hit TP
                if row['target'] == 2: return -1.0 # Hit SL
                return 0.0
            return 0.0
            
        # Use multipliers from labelling config for more realism
        # TP = 20 bps, SL = 10 bps
        tp_ret = 0.0020
        sl_ret = 0.0010
        fee = 0.0002 # 2 bps
        
        df_res['outcome'] = df_res.apply(calculate_pnl, axis=1)
        df_res['pnl_raw'] = df_res['outcome'].apply(lambda x: tp_ret if x == 1.0 else (-sl_ret if x == -1.0 else 0.0))
        
        # Apply fees only on entries
        df_res['pnl_net'] = df_res.apply(lambda x: x['pnl_raw'] - fee if x['signal'] != "NEUTRAL" else 0.0, axis=1)
        
        df_res['cum_pnl'] = df_res['pnl_net'].cumsum()
        
        # Metrics
        signals = df_res[df_res['signal'] != "NEUTRAL"]
        win_rate = (signals['outcome'] == 1.0).mean() if not signals.empty else 0
        total_trades = len(signals)
        total_pnl = df_res['pnl_net'].sum()
        
        report = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl_net": total_pnl,
            "profit_factor": abs(signals[signals['pnl_net'] > 0]['pnl_net'].sum() / signals[signals['pnl_net'] < 0]['pnl_net'].sum()) if any(signals['pnl_net'] < 0) else float('inf')
        }
        
        return report, df_res

def run_full_backtest():
    import yaml
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    # Detect environment: Cloud (/workspace) vs Local
    CLOUD_CONFIG = PROJECT_ROOT / "src" / "backtest" / "backtest_config_cloud.yaml"
    LOCAL_CONFIG = PROJECT_ROOT / "src" / "backtest" / "backtest_config.yaml"

    CONFIG_PATH = CLOUD_CONFIG if CLOUD_CONFIG.exists() and "/workspace" in str(PROJECT_ROOT) else LOCAL_CONFIG
    master_config_path = PROJECT_ROOT / "src/cloud/base_model/configs/master_config.yaml"
    
    with open(master_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        bt_config = yaml.safe_load(f)
    
    # Deep update master config with backtest specific overrides
    for key, value in bt_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    # ── [IMPORTANT] Override Model Paths for Backtest ──
    # The InferenceService uses relative paths from its models_local_dir.
    # We need to make sure it finds the models we specified.
    
    engine = BacktestEngine(config)
    labelled_dir = Path(config['pipeline_paths']['local_data_root']) / "labelled"
    files = sorted(list(labelled_dir.glob("*.parquet")))
    
    # Results directory handling
    results_dir = labelled_dir.parent / "backtest_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for pf in tqdm(files, desc="📅 Full Backtest Pipeline"):
        day_parquet = results_dir / f"res_{pf.stem}.parquet"
        
        # Resume Check
        if day_parquet.exists():
            logger.info(f"⏭️ Skipping {pf.name} (Result already exists).")
            # Load existing if needed for the final report
            # df_existing = pd.read_parquet(day_parquet)
            # all_results.extend(df_existing.to_dict('records'))
            continue
            
        res = engine.run_backtest_file(pf)
        if res:
            df_day = pd.DataFrame(res)
            # Save individual day
            df_day.to_parquet(day_parquet, index=False)
            logger.info(f"✅ Checkpoint: {day_parquet.name} saved.")
            all_results.extend(res)
            
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final Consolidation
    all_parquets = list(results_dir.glob("res_*.parquet"))
    if not all_parquets:
        logger.warning("❌ No results found to aggregate.")
        return

    logger.info(f"📚 Consolidating {len(all_parquets)} days into final report...")
    df_combined = pd.concat([pd.read_parquet(p) for p in all_parquets])
    
    report, _ = engine.analyze_performance(df_combined.to_dict('records'))
    
    logger.info("============== BACKTEST REPORT ==============")
    logger.info(json.dumps(report, indent=4))
    
    final_path = labelled_dir.parent / "backtest_results_COMPLETE.parquet"
    df_combined.to_parquet(final_path, index=False)
    logger.info(f"🏆 Backtest Complete! Final results in {final_path}")

if __name__ == "__main__":
    run_full_backtest()
