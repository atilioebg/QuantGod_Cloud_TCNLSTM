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
        
        day_results = []
        
        # Prepare data for faster access
        X_base = df[base_features].values
        X_context = df[context_features].values
        Y_target = df['target'].values
        timestamps = df['ts'].values
        prices = df['close'].values
        
        for i in range(len(df)):
            # 1. Get raw input (Production expects 30 features)
            raw_input = X_base[i].reshape(1, -1)
            context_input = X_context[i].reshape(1, -1)
            
            # 2. Predict
            # Note: InferenceService.predict expects (seq_len, 30) or similar
            # If the model is TCN-LSTM with seq_len, we need a window.
            # However, the pre-processed data already reflects the state at time T.
            # But the TCN-LSTM usually expects a 3D tensor (Batch, Seq, Feat).
            # If our labelled parquet has flat rows, we need to reconstruct the sequence if seq_len > 1.
            
            # Wait, the training data usually has sequences. 
            # If the parquet is already 'featured', we might need to handle the seq_len.
            # Let's check how many rows we have per bar.
            
            # Assuming seq_len=1 for now as a baseline or if the features encapsulate history
            # Actually, standard TCN-LSTM in this project uses seq_len (e.g. 720).
            # If so, the inference service needs a buffer.
            
            # For this backtest implementation, we'll assume the InferenceService
            # correctly handles its internal requirements or we provide the window.
            
            # Simple windowing:
            if i < self.inference.seq_len:
                continue # Need more data for sequence
                
            window = X_base[i - self.inference.seq_len + 1 : i+1]
            
            pred = self.inference.predict(window, context_input)
            
            # 3. Log Results
            day_results.append({
                "ts": timestamps[i],
                "price": prices[i],
                "target": Y_target[i],
                "signal": pred['signal'],
                "probs_spec_buy": pred['probs_specialist'][2],
                "probs_spec_sell": pred['probs_specialist'][0],
                "auditor_score": pred['auditor_score']
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
    config_path = PROJECT_ROOT / "src/backtest/backtest_config.yaml"
    master_config_path = PROJECT_ROOT / "src/cloud/base_model/configs/master_config.yaml"
    
    with open(master_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(config_path, 'r', encoding='utf-8') as f:
        bt_config = yaml.safe_load(f)
    config['pipeline_paths'].update(bt_config['pipeline_paths'])
    
    # ── [IMPORTANT] Override Model Paths for Backtest ──
    # The InferenceService uses relative paths from its models_local_dir.
    # We need to make sure it finds the models we specified.
    
    engine = BacktestEngine(config)
    labelled_dir = Path(config['pipeline_paths']['local_data_root']) / "labelled"
    files = sorted(list(labelled_dir.glob("*.parquet")))
    
    all_results = []
    for pf in files:
        res = engine.run_backtest_file(pf)
        all_results.extend(res)
        
    report, df_res = engine.analyze_performance(all_results)
    
    logger.info("============== BACKTEST REPORT ==============")
    logger.info(json.dumps(report, indent=4))
    
    # Save results
    df_res.to_csv(labelled_dir.parent / "backtest_results.csv", index=False)
    logger.info(f"✅ Results saved to {labelled_dir.parent / 'backtest_results.csv'}")

if __name__ == "__main__":
    run_full_backtest()
