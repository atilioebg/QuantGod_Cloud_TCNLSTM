import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Microstructure feature depth (top-N levels for OFI/Slope/RDI)
_FLOW_DEPTH = 5


class L2Transformer:
    def __init__(self, levels: int = 200, sampling_ms: int = 1000):
        self.levels = levels
        self.sampling_ms = sampling_ms
        self.bids_book: Dict[float, float] = {}
        self.asks_book: Dict[float, float] = {}
        self.last_sample_ts: int = -1

        # ── T-1 State (for dynamic flow features) ────────────────────────────
        self._prev_bids: list = []   # list of (price, size) tuples, top N sorted
        self._prev_asks: list = []   # list of (price, size) tuples, top N sorted
        self._prev_micro_price: float = np.nan
        self._prev_obi_l0: float = 0.0

    def reset_book(self):
        self.bids_book = {}
        self.asks_book = {}
        self.last_sample_ts = -1
        self._prev_bids = []
        self._prev_asks = []
        self._prev_micro_price = np.nan
        self._prev_obi_l0 = 0.0

    def process_message(self, msg: Dict) -> Optional[Dict]:
        """
        Processes a single L2 message (snapshot or delta) and returns a sampled row if interval reached.
        """
        msg_type = msg.get("type")
        ts = msg.get("ts")
        data = msg.get("data", {})

        if not ts: return None

        # 1. Update Orderbook
        if msg_type == "snapshot":
            self.bids_book = {float(p): float(s) for p, s in data.get("b", [])}
            self.asks_book = {float(p): float(s) for p, s in data.get("a", [])}
        else:
            # Deltas
            for p, s in data.get("b", []):
                price, size = float(p), float(s)
                if size == 0: self.bids_book.pop(price, None)
                else: self.bids_book[price] = size
            for p, s in data.get("a", []):
                price, size = float(p), float(s)
                if size == 0: self.asks_book.pop(price, None)
                else: self.asks_book[price] = size

        # 2. Temporal Sampling
        if self.last_sample_ts == -1 or ts - self.last_sample_ts >= self.sampling_ms:
            # Audit fix: if gap > 2 x sampling_ms, purge T-1 state (cross-file / data hole guard)
            if self.last_sample_ts != -1 and ts - self.last_sample_ts > 2 * self.sampling_ms:
                logger.debug(f"[transform] Timestamp gap detected ({ts - self.last_sample_ts}ms). Resetting T-1 state.")
                self._prev_bids = []
                self._prev_asks = []
                self._prev_micro_price = np.nan
                self._prev_obi_l0 = 0.0
            self.last_sample_ts = (ts // self.sampling_ms) * self.sampling_ms
            return self._capture_state(self.last_sample_ts)
        
        return None

    def _capture_state(self, ts: int) -> Dict:
        """Captures the top N levels and all dynamic microstructure features."""
        # Top Bids (Desc) / Asks (Asc)
        sorted_bids = sorted(self.bids_book.keys(), reverse=True)[:self.levels]
        sorted_asks = sorted(self.asks_book.keys())[:self.levels]

        # Cross-validation: Ensure no crosses
        if sorted_bids and sorted_asks:
            if sorted_bids[0] >= sorted_asks[0]:
                logger.warning(f"Orderbook crossed at {ts}: Bid {sorted_bids[0]} >= Ask {sorted_asks[0]}")

        row = {"ts": ts}

        # ── Level 0 base features ─────────────────────────────────────────────
        bid0_p = sorted_bids[0] if sorted_bids else np.nan
        bid0_s = self.bids_book[bid0_p] if sorted_bids else 0.0
        ask0_p = sorted_asks[0] if sorted_asks else np.nan
        ask0_s = self.asks_book[ask0_p] if sorted_asks else 0.0
        total_0 = bid0_s + ask0_s

        if sorted_bids and sorted_asks:
            micro_price = (bid0_p * ask0_s + ask0_p * bid0_s) / total_0 if total_0 > 0 else np.nan
            obi_l0 = (bid0_s - ask0_s) / total_0 if total_0 > 0 else 0.0
            row['spread'] = ask0_p - bid0_p  # Spread Intensity
        else:
            micro_price = np.nan
            obi_l0 = 0.0
            row['spread'] = np.nan

        row['micro_price'] = micro_price
        row['obi_l0'] = obi_l0

        # ── Top-N aggregates for OFI, Slope, RDI ─────────────────────────────
        n = _FLOW_DEPTH
        top_bids = [(p, self.bids_book[p]) for p in sorted_bids[:n]]
        top_asks = [(p, self.asks_book[p]) for p in sorted_asks[:n]]

        bid_vols = [s for _, s in top_bids]
        ask_vols = [s for _, s in top_asks]
        bid_vol_n = sum(bid_vols)
        ask_vol_n = sum(ask_vols)

        row['deep_obi_5'] = (bid_vol_n - ask_vol_n) / (bid_vol_n + ask_vol_n) if (bid_vol_n + ask_vol_n) > 0 else 0.0

        # ── Book Slope (Elasticity): ΔVol / ΔPrice for top N levels ──────────
        if len(top_bids) >= n and top_bids[0][0] != top_bids[-1][0]:
            row['bid_slope'] = bid_vol_n / abs(top_bids[0][0] - top_bids[-1][0])
        else:
            row['bid_slope'] = 0.0

        if len(top_asks) >= n and top_asks[0][0] != top_asks[-1][0]:
            row['ask_slope'] = ask_vol_n / abs(top_asks[-1][0] - top_asks[0][0])
        else:
            row['ask_slope'] = 0.0

        # ── Relative Depth Imbalance (RDI): Levels 1-4 vs Level 0 ────────────
        bid_depth_1_4 = sum(bid_vols[1:]) if len(bid_vols) > 1 else 0.0
        ask_depth_1_4 = sum(ask_vols[1:]) if len(ask_vols) > 1 else 0.0
        row['bid_rdi'] = (bid_depth_1_4 / bid0_s) if bid0_s > 0 else 0.0
        row['ask_rdi'] = (ask_depth_1_4 / ask0_s) if ask0_s > 0 else 0.0

        # ── Dynamic Flow Features (require T-1 state) ─────────────────────────
        prev_b = self._prev_bids
        prev_a = self._prev_asks

        if prev_b and prev_a:
            # OFI (Cont et al.) for each of the top N levels
            ofi_total = 0.0
            for i in range(n):
                # Bid side
                if i < len(top_bids) and i < len(prev_b):
                    cp, cs = top_bids[i]
                    pp, ps = prev_b[i]
                    if cp > pp:   ofi_bid_i = cs
                    elif cp == pp: ofi_bid_i = cs - ps
                    else:          ofi_bid_i = -ps
                else:
                    ofi_bid_i = 0.0

                # Ask side (inverted sign convention)
                if i < len(top_asks) and i < len(prev_a):
                    cp, cs = top_asks[i]
                    pp, ps = prev_a[i]
                    if cp < pp:   ofi_ask_i = cs
                    elif cp == pp: ofi_ask_i = cs - ps
                    else:          ofi_ask_i = -ps
                else:
                    ofi_ask_i = 0.0

                ofi_total += ofi_bid_i - ofi_ask_i

            row['ofi'] = ofi_total

            # MicroPrice Momentum: log-return of MicroPrice
            if not np.isnan(micro_price) and not np.isnan(self._prev_micro_price) and self._prev_micro_price > 0:
                row['micro_price_momentum'] = np.log(micro_price / self._prev_micro_price)
            else:
                row['micro_price_momentum'] = 0.0

            # Pressure Ratio: % velocity of OBI change
            prev_obi = self._prev_obi_l0
            if abs(prev_obi) > 1e-9:
                row['pressure_ratio'] = (obi_l0 - prev_obi) / abs(prev_obi)
            else:
                row['pressure_ratio'] = 0.0
        else:
            # First snapshot: zero-fill dynamic features (will be dropped by dropna if NaN)
            row['ofi'] = 0.0
            row['micro_price_momentum'] = 0.0
            row['pressure_ratio'] = 0.0

        # ── Update T-1 state ──────────────────────────────────────────────────
        self._prev_bids = top_bids
        self._prev_asks = top_asks
        self._prev_micro_price = micro_price
        self._prev_obi_l0 = obi_l0

        # ── Hard Cut: all 200 raw levels ──────────────────────────────────────
        for i in range(self.levels):
            if i < len(sorted_bids):
                p = sorted_bids[i]
                row[f"bid_{i}_p"] = p
                row[f"bid_{i}_s"] = self.bids_book[p]
            else:
                row[f"bid_{i}_p"] = np.nan
                row[f"bid_{i}_s"] = 0.0

            if i < len(sorted_asks):
                p = sorted_asks[i]
                row[f"ask_{i}_p"] = p
                row[f"ask_{i}_s"] = self.asks_book[p]
            else:
                row[f"ask_{i}_p"] = np.nan
                row[f"ask_{i}_s"] = 0.0

        return row

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies 1min resampling, log-returns and log-volume."""
        if df.empty: 
            logger.warning("apply_feature_engineering received an empty DataFrame")
            return df

        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # ── Resampling 1min ───────────────────────────────────────────────────
        # OFI is summed (net flow per minute), most others are averaged/last
        agg_map = {
            'micro_price': 'std',
            'spread': 'max',
            'obi_l0': 'mean',
            'deep_obi_5': 'mean',
            # New dynamic features
            'ofi': 'sum',                  # Total net flow per minute
            'micro_price_momentum': 'sum', # Cumulated log-return of micro-price
            'bid_slope': 'mean',
            'ask_slope': 'mean',
            'bid_rdi': 'mean',
            'ask_rdi': 'mean',
            'pressure_ratio': 'mean',
        }
        # Include all raw orderbook levels (using 'last' to represent best state at EOM)
        ob_cols_raw = [c for c in df.columns if ('bid_' in c or 'ask_' in c)
                       and not c.endswith(('_slope', '_rdi'))]
        for col in ob_cols_raw:
            agg_map[col] = 'last'

        resampled_others = df.resample('1min').agg(agg_map)

        # For Log Volume, we use tick count in the interval
        df['tick_count'] = 1
        resampled_vol = df['tick_count'].resample('1min').sum()

        resampled_ohlc = df['micro_price'].resample('1min').ohlc()
        final_df = pd.concat([resampled_ohlc, resampled_others, resampled_vol], axis=1)

        # ── Rename fixed aggregated columns ───────────────────────────────────
        # The ohlc comes first, then the agg_map columns in insertion order
        new_dynamic_cols = ['ofi', 'micro_price_momentum', 'bid_slope', 'ask_slope', 'bid_rdi', 'ask_rdi', 'pressure_ratio']
        agg_col_names = ['open', 'high', 'low', 'close', 'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi'] + new_dynamic_cols

        final_df.columns = agg_col_names + ob_cols_raw + ['tick_count']
        
        # ── Time-Aware Regularization (1440 MINUTE REINDEX) ───────────────────
        # Ensure perfect 1-minute continuity to avoid "teleportation" over missing API data
        try:
            # Anchor to the start of the day for the first timestamp in the file
            date_anchor = final_df.index[0].floor('D')
            full_idx_start = date_anchor
            full_idx_end = date_anchor.replace(hour=23, minute=59)
            full_idx = pd.date_range(start=full_idx_start, end=full_idx_end, freq='1min')
            
            # Reindex to full expected day
            final_df = final_df.reindex(full_idx)
            
            # 1. Flow Imputation (ZERO FILL)
            # If there's a gap, there is NO flow.
            # Do this BEFORE log_volume calculation to ensure log1p(0) = 0
            flow_features = ['ofi', 'micro_price_momentum', 'tick_count', 'pressure_ratio']
            if 'tick_count' in final_df.columns:
                final_df[flow_features] = final_df[flow_features].fillna(0)
            
            # 2. Price/Static Imputation (FORWARD FILL)
            # If there's a gap, orderbook state remains the same as the last seen
            price_state_features = [c for c in final_df.columns if c not in flow_features]
            final_df[price_state_features] = final_df[price_state_features].ffill()
            
            # For open/high/low in gap periods (where ffill propagated the LAST period's values),
            # logically, a gap minute should have open=high=low=close of the LAST close.
            # We enforce this constraint: if tick_count is 0, then open=high=low=close
            is_gap = final_df['tick_count'] == 0
            if is_gap.any():
                final_df.loc[is_gap, 'open'] = final_df.loc[is_gap, 'close']
                final_df.loc[is_gap, 'high'] = final_df.loc[is_gap, 'close']
                final_df.loc[is_gap, 'low'] = final_df.loc[is_gap, 'close']
                
        except Exception as e:
            logger.warning(f"[transform] Time-Aware reindexing failed: {e}. Falling back to dropna.")
            
        # Any residual NaNs at the very start of the day (before first trade) are dropped
        final_df.dropna(inplace=True)

        # Stationarity & Candle Shape
        prev_close = final_df['close'].shift(1)
        
        # Defragment dataframe before bulk insertions to prevent PerformanceWarning
        final_df = final_df.copy()
        
        # Core Candle Shape Features (Institutional Standard)
        # Body: Real movement within candle
        final_df['body'] = np.log(final_df['close'] / final_df['open'])
        # Wicks: Normalized by previous close to keep scale consistent
        final_df['upper_wick'] = (final_df['high'] - np.maximum(final_df['open'], final_df['close'])) / prev_close
        final_df['lower_wick'] = (np.minimum(final_df['open'], final_df['close']) - final_df['low']) / prev_close
        
        final_df['log_ret_close'] = np.log(final_df['close'] / prev_close)
        final_df['log_volume'] = np.log1p(final_df['tick_count'])

        # ── High Velocity "Sniper" Features (Acceleration) ────────────────────
        # OFI_delta_5: Change in net flow over the last 5 minutes
        final_df['ofi_delta_5'] = final_df['ofi'].diff(5)
        # RDI_delta_5: Change in relative depth imbalance over 5 minutes
        final_df['bid_rdi_delta_5'] = final_df['bid_rdi'].diff(5)
        final_df['ask_rdi_delta_5'] = final_df['ask_rdi'].diff(5)

        # ── Institutional Microstructure Features ───────────────────────────
        # 1. Micro-Price Delta: Momentum of the volume-weighted "true value"
        # We already have micro_price_momentum (summed 1min log-returns)
        # but let's add a 5min anchor for long-term drift.
        final_df['micro_price_delta_5'] = final_df['close'].pct_change(5) # 'close' in resampled is MicroPrice

        # 2. Book Asymmetry (L5): Total Volume imbalance in top 5 levels
        # Formula: log(sum_bids_5 / sum_asks_5)
        # We need the raw depth columns. We use the agg_map 'last' values.
        sum_bids_5 = sum(final_df[f"bid_{i}_s"] for i in range(5))
        sum_asks_5 = sum(final_df[f"ask_{i}_s"] for i in range(5))
        final_df['book_asymmetry_v5'] = np.log((sum_bids_5 + 1e-9) / (sum_asks_5 + 1e-9))

        # 3. Spread Z-Score (60min): Volatility and Liquidity Stress thermometer
        # Guard against div-by-zero using epsilon (1e-9) as requested.
        rolling_spread = final_df['max_spread'].rolling(window=60, min_periods=1)
        final_df['spread_zscore_60'] = (final_df['max_spread'] - rolling_spread.mean()) / (rolling_spread.std() + 1e-9)

        # 4. Volume Toxicity (V-PIN Lite): Flow toxicity vs total liquidity
        # Cumulative absolute OFI / Total depth over 5 minutes
        final_df['vpin_lite_5'] = final_df['ofi'].abs().rolling(5).sum() / (sum_bids_5 + sum_asks_5 + 1e-9)

        # ── Multi-Scale "Trigger" Features (1min Deltas) ────────────────────
        # Fast-reacting signals to compare against 5min contexts
        final_df['micro_price_delta_1'] = final_df['close'].pct_change(1)
        final_df['ofi_delta_1'] = final_df['ofi'].diff(1)
        final_df['bid_rdi_delta_1'] = final_df['bid_rdi'].diff(1)
        final_df['ask_rdi_delta_1'] = final_df['ask_rdi'].diff(1)

        # ── Phase 6: Orthogonal Deep-Book Features 🧬 ─────────────────────
        # 1. Kyle's Lambda: Resistance to flow (Price Impact)
        final_df['kyle_lambda'] = final_df['micro_price_delta_1'] / (final_df['ofi_delta_1'].abs() + 1e-9)

        # 2. Deep-to-Front Ratio (L200/L5): Structural intent vs short-term walls
        sum_bids_50_200 = sum(final_df[f"bid_{i}_s"] for i in range(50, 200))
        sum_asks_50_200 = sum(final_df[f"ask_{i}_s"] for i in range(50, 200))
        final_df['bid_deep_ratio'] = sum_bids_50_200 / (sum_bids_5 + 1e-9)
        final_df['ask_deep_ratio'] = sum_asks_50_200 / (sum_asks_5 + 1e-9)

        # 3. Book Convexity: Gradient between L1-L10 and L11-L20
        sum_bids_1_10 = sum(final_df[f"bid_{i}_s"] for i in range(1, 11))
        sum_bids_11_20 = sum(final_df[f"bid_{i}_s"] for i in range(11, 21))
        sum_asks_1_10 = sum(final_df[f"ask_{i}_s"] for i in range(1, 11))
        sum_asks_11_20 = sum(final_df[f"ask_{i}_s"] for i in range(11, 21))
        final_df['bid_convexity'] = sum_bids_1_10 / (sum_bids_11_20 + 1e-9)
        final_df['ask_convexity'] = sum_asks_1_10 / (sum_asks_11_20 + 1e-9)

        # Final cleanup for all rolling/diff features (NaNs to 0, Infs to 0)
        sniper_institutional_cols = [
            'ofi_delta_5', 'ofi_delta_1',
            'bid_rdi_delta_5', 'bid_rdi_delta_1', 
            'ask_rdi_delta_5', 'ask_rdi_delta_1',
            'micro_price_delta_5', 'micro_price_delta_1',
            'book_asymmetry_v5', 'spread_zscore_60', 'vpin_lite_5',
            'kyle_lambda', 'bid_deep_ratio', 'ask_deep_ratio', 'bid_convexity', 'ask_convexity'
        ]
        final_df[sniper_institutional_cols] = final_df[sniper_institutional_cols].replace([np.inf, -np.inf], 0).fillna(0)

        # ── Final Feature List (including Sniper, Institutional, Multi-Scale & Phase 6) 
        dynamic_features = [
            'ofi', 'ofi_delta_5', 'ofi_delta_1',
            'micro_price_momentum', 'micro_price_delta_5', 'micro_price_delta_1',
            'bid_slope', 'ask_slope', 
            'bid_rdi', 'bid_rdi_delta_5', 'bid_rdi_delta_1',
            'ask_rdi', 'ask_rdi_delta_5', 'ask_rdi_delta_1',
            'book_asymmetry_v5', 'spread_zscore_60', 'vpin_lite_5',
            'kyle_lambda', 'bid_deep_ratio', 'ask_deep_ratio', 'bid_convexity', 'ask_convexity',
            'pressure_ratio'
        ]
        agg_features = [
            'body', 'upper_wick', 'lower_wick', 'log_ret_close',
            'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
        ] + dynamic_features

        # Keep aggregated features + 'close' (label base) + raw orderbook levels
        # Fix: exclude processed sniper/institutional columns from raw ob_cols to prevent duplicates
        ob_cols = [c for c in final_df.columns if (('bid_' in c or 'ask_' in c) 
                   and not any(x in c for x in ['_slope', '_rdi', '_delta_', '_asymmetry', '_convexity']))]
        
        # Build final list and deduplicate while preserving order
        raw_final_cols = agg_features + ['close'] + ob_cols
        final_cols = []
        seen = set()
        for c in raw_final_cols:
            if c in final_df.columns and c not in seen:
                final_cols.append(c)
                seen.add(c)
        
        final_df = final_df[final_cols]

        # Log the new column set for traceability
        new_cols_present = [c for c in dynamic_features if c in final_df.columns]

        return final_df.dropna()

    def apply_zscore(self, df: pd.DataFrame, scaler_path: Optional[str] = None) -> pd.DataFrame:
        """
        Normalizes features.
        - Original OHLC/OBI features: StandardScaler (backward compatible with saved scalers).
        - New heavy-tailed flow features (OFI, Slope, etc.): RobustScaler to avoid
          extreme OFI spikes saturating TCN ReLU/GELU activations.
        """
        original_cols = [
            'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume', 'log_ret_close',
        ]
        flow_cols = [
            'ofi', 'ofi_delta_5', 'ofi_delta_1',
            'micro_price_momentum', 'micro_price_delta_5', 'micro_price_delta_1',
            'bid_slope', 'ask_slope', 
            'bid_rdi', 'bid_rdi_delta_5', 'bid_rdi_delta_1',
            'ask_rdi', 'ask_rdi_delta_5', 'ask_rdi_delta_1',
            'book_asymmetry_v5', 'spread_zscore_60', 'vpin_lite_5',
            'kyle_lambda', 'bid_deep_ratio', 'ask_deep_ratio', 'bid_convexity', 'ask_convexity',
            'pressure_ratio',
        ]
        original_cols = [c for c in original_cols if c in df.columns]
        flow_cols     = [c for c in flow_cols     if c in df.columns]

        if df.empty: return df

        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                scaler_bundle = pickle.load(f)
            # Support both old (single scaler) and new (dict of scalers) format
            if isinstance(scaler_bundle, dict):
                std_sc  = scaler_bundle['standard']
                rob_sc  = scaler_bundle['robust']
                if original_cols: df[original_cols] = std_sc.transform(df[original_cols])
                if flow_cols:     df[flow_cols]     = rob_sc.transform(df[flow_cols])
            else:
                # Legacy: single StandardScaler — apply only to original cols
                all_legacy = [c for c in original_cols if c in df.columns]
                if all_legacy: df[all_legacy] = scaler_bundle.transform(df[all_legacy])
        else:
            from sklearn.preprocessing import StandardScaler, RobustScaler
            std_sc = StandardScaler()
            rob_sc = RobustScaler()
            if original_cols: df[original_cols] = std_sc.fit_transform(df[original_cols])
            if flow_cols:     df[flow_cols]     = rob_sc.fit_transform(df[flow_cols])
            if scaler_path:
                Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump({'standard': std_sc, 'robust': rob_sc}, f)

        return df
