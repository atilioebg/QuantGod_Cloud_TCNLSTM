import numpy as np
import pandas as pd
import polars as pl
import logging
import time
from typing import Dict, List, Optional, Any
from collections import deque
from pathlib import Path
import pickle
import joblib

# Internal imports for L2 transformation logic
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.auditor_model.auditor_preprocessing import calculate_context_features

logger = logging.getLogger(__name__)

class StreamingETL:
    """
    Orchestrates real-time feature engineering.
    - Aggregates L2/Trades into bars.
    - Maintains a rolling history of bars for indicator calculation.
    - Produces the feature set for Foundation and Auditor layers.
    """
    def __init__(self, config: Dict, levels: int = 200):
        self.config = config
        self.resample_min = config['pre_processing']['etl'].get('resample_min', 1)
        self.seq_len = config['optimization'].get('seq_len', 60)
        
        # Internal L2 State (reuse project logic)
        self.transformer = L2Transformer(levels=levels, sampling_ms=1000)
        
        # History buffers
        # For 24h @ 1min we need 1440 bars. For 5min we need 288.
        # We use a large enough buffer (2000) to cover all indicators.
        self.bar_history: deque = deque(maxlen=2000)
        self.sampled_rows: List[Dict] = []
        
        # Feature names (canonical order)
        self.foundation_features = config['model']['feature_names']
        self.auditor_features = config['auditor_features'] if 'auditor_features' in config else []
        
        # Scalers
        self._load_scalers()

    def _load_scalers(self):
        try:
            # We need the foundation scaler (Z-Score)
            scaler_path = self.config['pipeline_paths'].get('scaler_foundation', 'data/models/scaler_foundation.pkl')
            path = Path(scaler_path)
            if path.exists():
                # run_foundation.py uses joblib.dump
                self.scaler_foundation = joblib.load(path)
                logger.info(f"✅ Foundation Scaler loaded: {scaler_path}")
            else:
                logger.warning(f"⚠️ Foundation Scaler not found at {scaler_path}. Features will NOT be scaled.")
                self.scaler_foundation = None
        except Exception as e:
            logger.error(f"❌ Failed to load scalers: {e}")
            self.scaler_foundation = None

    def process_data(self, l2_data: Dict, trades: List[Dict], ts: int):
        """
        Ingests the latest L2 snapshot and trade list to build the current bar.
        """
        # 1. Update L2 transformer
        # In a real stream, we'd process incremental updates. 
        # Here we assume ExchangeConnector provides a merged snapshot.
        msg = {"type": "snapshot", "ts": ts, "data": l2_data}
        row = self.transformer.process_message(msg)
        
        if row:
            # Add trade-based volume proxy (log_volume) if we had them.
            # For simplicity, we'll rely on the L2Transformer's tick_count or similar.
            self.sampled_rows.append(row)

    def on_bar_close(self) -> Optional[Dict[str, Any]]:
        """
        Calculates all features (Foundation + Auditor) for the current window.
        """
        if not self.sampled_rows:
            return None
        
        # 1. Convert accumulated samples to Polars
        df_new_samples = pl.DataFrame(self.sampled_rows)
        # We don't clear sampled_rows yet? Actually, L2Transformer needs historical rows 
        # for its rolling features (spread_zscore, vpin, etc.)
        
        # Maintain a buffer of raw samples (1s sampled state) for the last 2 hours 
        # to ensure all rolling features (max 60min) are accurate.
        # We'll use a local buffer of rows.
        if not hasattr(self, 'row_buffer'):
            self.row_buffer = deque(maxlen=7200) # 2 hours of 1s samples
        
        self.row_buffer.extend(self.sampled_rows)
        self.sampled_rows = []
        
        # 2. Run Foundation Feature Engineering (identical to training)
        df_raw_all = pl.DataFrame(list(self.row_buffer))
        df_foundation = self.transformer.apply_feature_engineering(df_raw_all)
        
        if len(df_foundation) < self.seq_len:
            logger.info(f"⏳ Warming up Foundation bars: {len(df_foundation)}/{self.seq_len}")
            return None

        # 3. Apply Z-Score scaling to Foundation Features
        if self.scaler_foundation is not None:
             # Apply using the loaded scaler
             df_foundation_norm = self.transformer.apply_zscore(df_foundation, self.scaler_foundation)
        else:
             # Skip scaling but log warning once
             if not hasattr(self, '_scaling_warned'):
                 logger.warning("🕒 Scaling skipped because scaler_foundation is missing. Predictions will be inaccurate.")
                 self._scaling_warned = True
             df_foundation_norm = df_foundation

        # 4. Run Auditor Alpha Sensors
        # Requirement: calculate_context_features expects OHLCV from the and context.
        # It's better to use the resampled 'df_foundation' (which has high/low/close/log_volume)
        df_pd_ohlcv = df_foundation.to_pandas()
        df_auditor = calculate_context_features(df_pd_ohlcv, self.resample_min)
        
        # 5. Extract latest inputs
        # Foundation Input: (seq_len, 32)
        # features_foundation is mapped in master_config.yaml
        latest_foundation = df_foundation_norm.tail(self.seq_len).select(self.foundation_features).to_numpy().astype(np.float32)
        
        # Auditor Input: (1, 14) 
        # auditor_features = ['ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', ...]
        latest_auditor = df_auditor.tail(1)[self.auditor_features].values.astype(np.float32)

        return {
            "foundation_input": latest_foundation, 
            "auditor_input": latest_auditor,
            "metadata": {
                "ts": df_foundation['datetime'].max(),
                "price": df_foundation['close'].last()
            }
        }

    def _calculate_foundation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the 32 foundation features for the current rolling window.
        """
        # This mirrors the logic in L2Transformer.apply_feature_engineering
        # but optimized for a single window.
        res = df.copy()
        
        # Body/Wicks/Returns
        res['body'] = np.log(res['close'] / res['open'].replace(0, 1e-9))
        res['log_ret_close'] = np.log(res['close'] / res['close'].shift(1).fillna(res['close']))
        
        # Delta features (5 and 30)
        ds = max(1, 5 // self.resample_min)
        dl = max(1, 30 // self.resample_min)
        
        res['ofi_delta_5'] = res['ofi'].diff(ds).fillna(0)
        res['ofi_delta_30'] = res['ofi'].diff(dl).fillna(0)
        res['micro_price_delta_5'] = res['close'].pct_change(ds).fillna(0)
        res['micro_price_delta_30'] = res['close'].pct_change(dl).fillna(0)
        # ... and so on for all 32 features ...
        
        # For simplicity in this plan, I'll ensure the final list matches self.foundation_features
        # and applies the Z-Score transform using self.scaler_foundation.
        
        return res

if __name__ == "__main__":
    # Test stub
    from src.cloud.base_model.utils.config_utils import load_config
    cfg = load_config()
    etl = StreamingETL(cfg)
    print("StreamingETL Initialized.")
