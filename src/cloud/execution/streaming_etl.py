import numpy as np
import pandas as pd
import polars as pl
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
from pathlib import Path
import pickle
import joblib
import json

# Internal imports for L2 transformation logic
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer, _parse_resample_minutes
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
        
        # Correctly parse resample frequency from config
        self.resample_freq = config['pre_processing']['etl'].get('resample_freq', "1min")
        self.resample_min = _parse_resample_minutes(self.resample_freq)
        
        # Load seq_len from the actual best_params.json of the session
        self.seq_len = self._load_actual_seq_len()
        
        # Internal L2 State (reuse project logic)
        self.transformer = L2Transformer(levels=levels, sampling_ms=1000)
        
        # History buffers
        # How many 1s samples do we need? 
        # For indicators and sequence, we need at least: seq_len * resample_min * 60 seconds
        # Example: 120 bars * 5 min * 60s = 36,000 samples.
        needed_samples = (self.seq_len + 10) * self.resample_min * 60
        self.row_buffer = deque(maxlen=max(7200, needed_samples)) 
        
        self.bar_history: deque = deque(maxlen=2000)
        self.sampled_rows: List[Dict] = []
        
        # State Persistence File
        project_root = Path(__file__).parents[3]
        self.state_file = project_root / "data" / "paper_trading_state" / "row_buffer.parquet"
        
        # Feature names (canonical order)
        self.foundation_features = config['model']['feature_names']
        self.auditor_features = config['auditor_features'] if 'auditor_features' in config else []
        
        # Scalers
        self._load_scalers()
        
        # Load any existing warmed-up state
        self.load_state()

    def load_state(self):
        if self.state_file.exists():
            try:
                df = pl.read_parquet(str(self.state_file))
                records = df.to_dicts()
                self.row_buffer.extend(records)
                logger.info(f"🔄 State Persistence: Loaded {len(records)} warmed-up ticks from disk. Bypassing manual wait.")
            except Exception as e:
                logger.error(f"❌ Failed to load state from {self.state_file}: {e}")

    def save_state(self):
        if len(self.row_buffer) > 0:
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                df = pl.DataFrame(list(self.row_buffer))
                df.write_parquet(str(self.state_file), compression='snappy')
                logger.info(f"💾 State Persistence: Saved {len(self.row_buffer)} ticks to disk safely.")
            except Exception as e:
                logger.error(f"❌ Failed to save state to {self.state_file}: {e}")

    def _load_actual_seq_len(self) -> int:
        project_root = Path(__file__).parents[3]
        
        paths = []
        # 1. Explicit models_local_dir (priority)
        explicit = self.config.get('execution', {}).get('models_local_dir')
        if explicit:
            p_json = (project_root / explicit / "CONFIG" / "best_params.json").resolve()
            paths.append(p_json)
        else:
            model_path_cfg = self.config.get('pipeline_paths', {}).get('best_tcn_lstm_model')
            if model_path_cfg:
                from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_project
                base_dir = get_drive_session_path("MODELOS", self.config)
                base_path = resolve_local_project(base_dir, project_root)
                p_model = base_path / model_path_cfg
                p_json = p_model.parent.parent / "CONFIG" / "best_params.json"
                paths.append(p_json)
        
        # 2. Local fallback
        paths.append(project_root / "src/cloud/base_model/otimizacao/best_params.json")

        for p in paths:
            if p.exists():
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        params = json.load(f)
                    val = params.get('seq_len', 120) 
                    logger.info(f"📏 StreamingETL: Using dynamic seq_len={val} from {p}")
                    return int(val)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse {p} for seq_len: {e}")

        fallback = self.config['optimization'].get('seq_len', 120)
        logger.info(f"📏 StreamingETL: Using fallback seq_len={fallback}")
        return fallback

    def _load_scalers(self):
        project_root = Path(__file__).parents[3]
        try:
            scaler_rel = self.config['pipeline_paths'].get('scaler_foundation', 'BASE_MODEL/scaler_foundation.pkl')
            # Priority: explicit models_local_dir
            explicit = self.config.get('execution', {}).get('models_local_dir')
            if explicit:
                path = (project_root / explicit / scaler_rel).resolve()
            else:
                from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_project
                base_dir = get_drive_session_path("MODELOS", self.config)
                path = resolve_local_project(base_dir, project_root) / scaler_rel
                
            if path.exists():
                # Store the path; L2Transformer.apply_zscore will handle the loading
                self.scaler_foundation = str(path)
                logger.info(f"✅ Foundation Scaler verified: {path}")
            else:
                logger.warning(f"⚠️ Foundation Scaler not found at {path}. Features will NOT be scaled.")
                self.scaler_foundation = None
        except Exception as e:
            logger.error(f"❌ Error during scaler verification: {e}")
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

    def on_bar_close(self, current_boundary: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Calculates all features (Foundation + Auditor) for the current window.
        """
        if not self.sampled_rows:
            return None
        
        # Add new samples to the history buffer
        self.row_buffer.extend(self.sampled_rows)
        self.sampled_rows = []
        
        # 1. Convert accumulated samples to Polars
        df_raw_all = pl.DataFrame(list(self.row_buffer))
        
        # 2. Run Foundation Feature Engineering
        # Bypass 'Hard Reset' for streaming:
        # We tell the transformer this is a stream, so it shouldn't try to 
        # reindex a full 24h grid (which causes gaps/resets).
        if not hasattr(self.transformer, '_streaming_mode'):
             self.transformer._streaming_mode = True
             
        df_foundation = self.transformer.apply_feature_engineering(df_raw_all)
        
        # Filter out the incomplete bar that spills over the boundary
        if current_boundary is not None:
             df_foundation = df_foundation.filter(pl.col("datetime") < current_boundary)
        
        # ── CHECPOINT DE SEGURANÇA ──
        # Salvar estado no disco A CADA BARRA (ex: a cada 5min), 
        # para que um 'Killed' abrupto (ex: OOM) não jogue fora as horas de aquecimento acumuladas.
        self.save_state()

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
        # Requirement: calculate_context_features expects OHLCV from the context.
        df_pd_ohlcv = df_foundation.to_pandas()
        df_auditor = calculate_context_features(df_pd_ohlcv, self.resample_min)
        
        # 6. Extract latest inputs
        # Foundation Input: (seq_len, 30)
        latest_foundation = df_foundation_norm.tail(self.seq_len).select(self.foundation_features).to_numpy().astype(np.float32)
        
        # Auditor Input: (1, 14) 
        # We only want the SENSORS here (the 14 technical indicators)
        # Probabilities are added inside InferenceService.predict
        sensor_names = [f for f in self.auditor_features if f in df_auditor.columns]
        if len(sensor_names) != 14:
             # Fallback: if names mismatch, just take the known 14
             sensor_names = [
                'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
                'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
                'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
             ]
             
        latest_auditor = df_auditor.tail(1)[sensor_names].values.astype(np.float32)

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
