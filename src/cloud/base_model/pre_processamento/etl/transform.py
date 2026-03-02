import json
import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, List, Optional
import pickle
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

# ── ETL config loader ───────────────────────────────────────────
def _parse_resample_minutes(freq: str) -> int:
    """
    Converts a resample frequency string (e.g. '1min', '5min') to integer minutes.
    Supports Pandas offset aliases: min, T, h, H.
    """
    freq = freq.strip()
    for suffix in ('min', 'T', 'Min'):
        if freq.endswith(suffix):
            return max(1, int(freq.replace(suffix, '')))
    if freq.endswith(('h', 'H')):
        return int(freq[:-1]) * 60
    return 1  # safe fallback: assume 1 minute


def _load_etl_config() -> dict:
    """
    Loads all ETL parameters from master_config.yaml at import time and converts
    window parameters from real-minutes (*_min) to bars (dividing by resample_freq).

    This ensures every indicator maintains its intended temporal meaning regardless
    of which resample_freq is active. Example:
      spread_zscore_window_min=60 + resample_freq='1min'  -> 60 bars
      spread_zscore_window_min=60 + resample_freq='5min'  -> 12 bars  (backward-compatible)

    Falls back to safe defaults expressed in minutes when the config is unavailable.
    """
    _DEFAULT_MINS = {
        "spread_zscore_window_min": 60,
        "vpin_window_min":           25,
        "delta_short_min":            5,
        "delta_long_min":            30,
    }
    _DEFAULTS = {
        "flow_depth":            5,
        "resample_freq":         "5min",
        "book_asymmetry_depth":  5,
        "deep_book_start":       50,
        "convexity_near_end":    10,
        "convexity_far_end":     20,
    }
    cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        etl = cfg["pre_processing"]["etl"]

        resample_freq = etl.get("resample_freq", _DEFAULTS["resample_freq"])
        resample_min  = _parse_resample_minutes(resample_freq)

        resolved = {k: etl.get(k, v) for k, v in _DEFAULTS.items()}
        resolved["resample_freq"] = resample_freq
        resolved["resample_min"]  = resample_min

        for min_key, default_min in _DEFAULT_MINS.items():
            real_minutes      = int(etl.get(min_key, default_min))
            bar_key           = min_key.replace("_min", "")
            resolved[bar_key] = max(1, real_minutes // resample_min)
            resolved[min_key] = real_minutes

        # Explicitly include nesting for clipping and audit
        resolved["clipping"] = etl.get("clipping", {"enabled": False})
        resolved["audit"] = etl.get("audit", {"generate_report": False})

        return resolved
    except Exception:
        resample_min = _parse_resample_minutes(_DEFAULTS["resample_freq"])
        fallback = dict(_DEFAULTS)
        fallback["resample_min"] = resample_min
        for min_key, default_min in _DEFAULT_MINS.items():
            bar_key           = min_key.replace("_min", "")
            fallback[bar_key] = max(1, default_min // resample_min)
            fallback[min_key] = default_min
            
        # Add clipping config to fallback if needed
        fallback["clipping"] = {"enabled": False}
        return fallback


_ETL_CFG = _load_etl_config()
# Module-level shortcut kept for backward compatibility
_FLOW_DEPTH = _ETL_CFG["flow_depth"]


class L2Transformer:
    def __init__(self, levels: int = 200, sampling_ms: int = 1000,
                 flow_depth: int = None, etl_cfg: dict = None):
        """
        Args:
            levels:      Number of orderbook levels to capture per snapshot.
            sampling_ms: Snapshot interval in milliseconds.
            flow_depth:  Override for top-N levels used in OFI/Slope/RDI.
                         Defaults to master_config.yaml → pre_processing.etl.flow_depth.
            etl_cfg:     Full ETL config dict (all parameters). If None, loads
                         from master_config.yaml automatically.
        """
        # Merge caller overrides on top of the module-level config
        cfg = dict(_ETL_CFG)  # copy
        if etl_cfg:
            cfg.update(etl_cfg)

        self.levels = levels
        self.sampling_ms = sampling_ms
        self.flow_depth  = flow_depth if flow_depth is not None else cfg["flow_depth"]

        # Store all ETL parameters for use inside apply_feature_engineering
        self._resample_freq        = cfg["resample_freq"]
        self._resample_min         = int(cfg["resample_min"])          # minutes per bar (e.g. 1 or 5)
        self._spread_zscore_window = int(cfg["spread_zscore_window"])  # in bars, resolved from *_min
        self._vpin_window          = int(cfg["vpin_window"])           # in bars, resolved from *_min
        self._vpin_window_min      = int(cfg.get("vpin_window_min", 25))
        self._delta_short          = int(cfg["delta_short"])           # in bars, resolved from *_min
        self._delta_long           = int(cfg["delta_long"])            # in bars, resolved from *_min
        # Real-minute values kept for column-name labels (e.g. ofi_delta_5, ofi_delta_30)
        self._delta_short_min      = int(cfg["delta_short_min"])       # real minutes (e.g. 5)
        self._delta_long_min       = int(cfg["delta_long_min"])        # real minutes (e.g. 30)
        self._book_asym_depth      = int(cfg["book_asymmetry_depth"])
        self._deep_book_start      = int(cfg["deep_book_start"])
        self._convexity_near_end   = int(cfg["convexity_near_end"])
        self._convexity_far_end    = int(cfg["convexity_far_end"])
        
        self._clipping_cfg = cfg.get("clipping", {"enabled": False})
        self._etl_cfg = cfg # Store full cfg for healing rules
        self.audit_report = {
            "file_id": "unknown",
            "clipping_events": {}, # feat -> {count, max_original}
            "outlier_density": 0.0,
            "temporal_gaps": [],
            "max_gap_before": 0.0,
            "max_gap_after": 0.0,
            "healed": False,
            "healing_details": [],
            "features_healed": [],
            "num_islands_generated": 0,
            "total_rows_retained": 0,
            "gap_fragmentation_events": []
        }

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
        self.audit_report = {
            "file_id": self.audit_report.get("file_id", "unknown"),
            "clipping_events": {},
            "outlier_density": 0.0,
            "temporal_gaps": [],
            "max_gap_before": 0.0,
            "max_gap_after": 0.0,
            "healed": False,
            "healing_details": [],
            "features_healed": [],
            "num_islands_generated": 0,
            "total_rows_retained": 0,
            "gap_fragmentation_events": []
        }

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
        n = self.flow_depth
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

    def _apply_soft_clipping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Winsorization (clipping) based on P99 * multiplier.
        Tracks changes for the quality report.
        """
        if not self._clipping_cfg.get("enabled", False):
            return df

        target_cols = self._clipping_cfg.get("target_columns", [])
        multiplier = self._clipping_cfg.get("p99_multiplier", 10)
        noise_floor = self._clipping_cfg.get("noise_floor", 1e-6)
        
        total_cells = df.shape[0] * len(target_cols) if target_cols else 1
        clipped_count = 0
        
        for col in target_cols:
            if col not in df.columns:
                continue
            
            p99 = df[col].quantile(0.99)
            # Skip if P99 is negligible (noise floor)
            if pd.isna(p99) or p99 < noise_floor:
                continue
                
            threshold = p99 * multiplier
            outliers_mask = df[col] > threshold
            
            if outliers_mask.any():
                max_val = df[col].max()
                count = outliers_mask.sum()
                clipped_count += count
                
                # Store for audit
                self.audit_report["clipping_events"][col] = {
                    "count": int(count),
                    "max_original": float(max_val),
                    "threshold": float(threshold)
                }
                
                # Dynamic format: use scientific if value is very small
                fmt_max = ".4e" if max_val < 0.01 else ".4f"
                fmt_thr = ".4e" if threshold < 0.01 else ".4f"
                logger.warning(f"☢️ [CLIPPING] {self.audit_report['file_id']}: {col} max ({max_val:{fmt_max}}) reduced to {threshold:{fmt_thr}} (10x P99)")
                df.loc[outliers_mask, col] = threshold
                
        self.audit_report["outlier_density"] = (clipped_count / total_cells) * 100 if total_cells > 0 else 0.0
        return df

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies temporal resampling (resample_freq), log-returns and log-volume.
        All parameters are driven by master_config.yaml → pre_processing.etl.
        """
        if df.empty:
            logger.warning("apply_feature_engineering received an empty DataFrame")
            return df

        freq   = self._resample_freq
        ds     = self._delta_short          # bars
        dl     = self._delta_long           # bars
        ds_lbl = str(self._delta_short_min) # label = real minutes (e.g. "5")
        dl_lbl = str(self._delta_long_min)  # label = real minutes (e.g. "30")

        df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)

        # ── Resampling (freq from config) ─────────────────────────────────────
        # OFI is summed (net flow per bar), most others are averaged/last
        agg_map = {
            'micro_price': 'std',
            'spread': 'max',
            'obi_l0': 'mean',
            'deep_obi_5': 'mean',
            'ofi': 'sum',
            'micro_price_momentum': 'sum',
            'bid_slope': 'mean',
            'ask_slope': 'mean',
            'bid_rdi': 'mean',
            'ask_rdi': 'mean',
            'pressure_ratio': 'mean',
        }
        ob_cols_raw = [c for c in df.columns if ('bid_' in c or 'ask_' in c)
                       and not c.endswith(('_slope', '_rdi'))]
        for col in ob_cols_raw:
            agg_map[col] = 'last'

        resampled_others = df.resample(freq).agg(agg_map)
        df['tick_count'] = 1
        resampled_vol  = df['tick_count'].resample(freq).sum()
        resampled_ohlc = df['micro_price'].resample(freq).ohlc()
        final_df = pd.concat([resampled_ohlc, resampled_others, resampled_vol], axis=1)

        # ── Rename columns ─────────────────────────────────────────────────────
        new_dynamic_cols = ['ofi', 'micro_price_momentum', 'bid_slope', 'ask_slope',
                            'bid_rdi', 'ask_rdi', 'pressure_ratio']
        agg_col_names = ['open', 'high', 'low', 'close',
                         'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi'] + new_dynamic_cols
        final_df.columns = agg_col_names + ob_cols_raw + ['tick_count']

        # ── Time-Aware Regularization (full-day reindex at resample_freq) ────────
        try:
            # Anchor to start of day
            # v4.8.6: Robust Daily Anchoring (GMT-3 / 21:00 UTC Resilience)
            # Instead of mode(), we use the first timestamp with a 4h-forward normalization.
            # Files starting >= 20:00 UTC (like GMT-3 dumps) are anchored to the NEXT day.
            # Files starting early (00:00 - 19:59 UTC) are anchored to CURRENT day.
            first_ts_utc = final_df.index[0]
            if first_ts_utc.tzinfo is None:
                first_ts_utc = first_ts_utc.tz_localize('UTC')
            
            if first_ts_utc.hour >= 20:
                # 21:00 UTC -> +4h = 01:00 Next Day -> normalize = 00:00 Next Day
                date_anchor = (first_ts_utc + pd.Timedelta(hours=4)).normalize()
            else:
                # 00:00 UTC -> normalize = 00:00 Current Day
                date_anchor = first_ts_utc.normalize()
            
            if date_anchor.tzinfo is None:
                date_anchor = date_anchor.tz_localize('UTC')
            else:
                date_anchor = date_anchor.tz_convert('UTC')
            
            # Robust freq_min calculation (v4.7 Gold)
            try:
                freq_offset = pd.tseries.frequencies.to_offset(freq)
                freq_min = int(pd.to_timedelta(freq).total_seconds() // 60)
            except:
                freq_min = 1 # Fallback
            
            periods = (24 * 60) // freq_min
            # Ensure full_idx is strictly UTC normalized
            full_idx = pd.date_range(start=date_anchor, periods=periods, freq=freq, tz='UTC')
            
            # Reindex fills missing minutes with NaNs
            final_df = final_df.reindex(full_idx)
            
            # Identify gaps (Silence or NaN)
            # Critical: after reindex, missing bars are NaNs. Consecutive zero-trades are 0.
            is_gap = (final_df['tick_count'].isna()) | (final_df['tick_count'] == 0)
            
            # Pre-healing gap audit (BEFORE any filling)
            # v4.8 Gold: Boundary-Aware Gap Detection
            if is_gap.any():
                valid_indices = final_df.index[~is_gap]
                if not valid_indices.empty:
                    # 1. Internal gaps
                    diffs_pre = valid_indices.to_series().diff().dropna()
                    max_internal = float(diffs_pre.max().total_seconds() / 60) if not diffs_pre.empty else 0.0
                    
                    # 2. Boundary gaps (v4.8)
                    # Gap from start of day to first valid
                    gap_start = float((valid_indices[0] - full_idx[0]).total_seconds() / 60)
                    # Gap from last valid to end of day
                    gap_end = float((full_idx[-1] - valid_indices[-1]).total_seconds() / 60)
                    
                    # We subtract 1 freq to get the actual "blackout" time if it's an internal gap
                    # For external gaps (start/end), we take the raw duration
                    self.audit_report["max_gap_before"] = max(max_internal, gap_start, gap_end)
                else:
                    self.audit_report["max_gap_before"] = 1440.0
            else:
                self.audit_report["max_gap_before"] = 0.0

            logger.info(f"💓 Heartbeat [Pre-Healing]: {len(final_df)} rows | Max Gap: {self.audit_report['max_gap_before']}m")
            logger.info(f"💓 Heartbeat [Trace]: idx[0]={final_df.index[0]}, anchor={date_anchor}, full[0]={full_idx[0]}")

            # Group consecutive gaps
            gap_groups = (is_gap != is_gap.shift()).cumsum()
            gaps = is_gap[is_gap].groupby(gap_groups[is_gap])
            
            # Define healing groups
            group_a_cols = ['open', 'high', 'low', 'close', 'max_spread', 'volatility'] + ob_cols_raw
            group_a_cols = [c for c in group_a_cols if c in final_df.columns]
            
            group_b_base = ['mean_obi', 'mean_deep_obi', 'bid_slope', 'ask_slope', 'bid_rdi', 'ask_rdi', 'pressure_ratio']
            group_b_cols = [c for c in group_b_base if c in final_df.columns]
            
            flow_cols_base = ['tick_count', 'ofi', 'micro_price_momentum', 'volatility']
            actual_flow_cols = [c for c in flow_cols_base if c in final_df.columns]
            group_c_cols = actual_flow_cols # Unified flow columns for Zero-fill

            heal_threshold_min = self._etl_cfg.get("healing", {}).get("max_gap_minutes", 5)
            fragment_threshold_min = 30.0 # Sniper Gold Hard Reset
            
            # --- Island Management (Island Split v4.6 Protocol) ---
            # v4.8.3: Force island_id column BEFORE gap logic to guarantee presence
            final_df = final_df.copy() # Defragment after reindexing
            final_df['island_id'] = 0
            current_island_id = 0
            unhealed_mask = is_gap.copy()
            healed_any = False

            for _, group in gaps:
                gap_len_min = len(group) * freq_min
                
                # 1. ALWAYS Apply Group B (Median) and Group C (Zero) to preserve the bars audit-trail
                if not group.empty:
                    # Robust assignment for Median (Group B)
                    median_vals = final_df[group_b_cols].median()
                    for b_col in group_b_cols:
                        final_df.loc[group.index, b_col] = median_vals[b_col]
                    
                    # Robust assignment for Zero-fill (Group C)
                    for c_col in group_c_cols:
                        final_df.loc[group.index, c_col] = 0.0

                # 2. Island Split Trigger (Hard Reset > 30 min)
                if gap_len_min > fragment_threshold_min:
                    current_island_id += 1
                    # New island starts AFTER the gap
                    start_of_new = group.index[-1] + freq_offset
                    final_df.loc[start_of_new:, 'island_id'] = current_island_id
                    
                    self.audit_report["gap_fragmentation_events"].append(str(group.index[0]))
                    logger.warning(f"🏝️ [ISLAND SPLIT] Hard Reset at {group.index[0]} due to {gap_len_min}min gap. New Island ID: {current_island_id}")
                    unhealed_mask.loc[group.index] = True 
                
                # 3. Level 1 Healing (Short gaps <= 5 min)
                elif gap_len_min <= heal_threshold_min:
                    healed_any = True
                    self.audit_report["healed"] = True
                    self.audit_report["healing_details"].append(f"Healed {gap_len_min}min gap at {group.index[0]}")
                    unhealed_mask.loc[group.index] = False # Healed bars are NO LONGER gaps
                
                # 4. Dangerous Zone (5 min < gap <= 30 min)
                else:
                    unhealed_mask.loc[group.index] = True

            # Apply Linear Interpolation to Group A ONLY for healed gaps (short limit)
            interp_limit = int(heal_threshold_min / freq_min)
            final_df[group_a_cols] = final_df[group_a_cols].interpolate(method='linear', limit=interp_limit)

            # ── SELECTIVE FILLING (FFILL) Grouped by Island ──────────────────
            # This is critical: forward fill MUST NOT cross island boundaries.
            # We use transform to apply ffill per group to keep the original index
            state_cols = [c for c in final_df.columns if c not in actual_flow_cols and c != 'island_id']
            # bfill/ffill per island to avoid leakage
            for col in state_cols:
                final_df[col] = final_df.groupby('island_id')[col].transform(lambda x: x.ffill().bfill())

            if healed_any:
                self.audit_report["features_healed"] = ["Group A: Linear", "Group B: Median", "Group C: Zero"]
                logger.info(f"🩹 [HEALING] Level 1 restored short gaps in {self.audit_report['file_id']}")

            # Fix OHLC logic for gaps (High/Low = Close)
            if unhealed_mask.any():
                # Direct assignment of series to avoid ndarray reshape ValueError
                final_df.loc[unhealed_mask, 'open'] = final_df.loc[unhealed_mask, 'close']
                final_df.loc[unhealed_mask, 'high'] = final_df.loc[unhealed_mask, 'close']
                final_df.loc[unhealed_mask, 'low']  = final_df.loc[unhealed_mask, 'close']

            # Post-healing gap audit (Final check)
            # v4.8 Gold: Ghost Gap Prevention (ignore trailing gaps if island split)
            if unhealed_mask.any():
                # We only count gaps that are NOT just trailing gaps after the last valid row in final_df
                # Because final_df.dropna() will remove them anyway.
                valid_mask = ~final_df.isnull().any(axis=1)
                if valid_mask.any():
                    last_valid_idx = final_df.index[valid_mask][-1]
                    effective_unhealed = unhealed_mask.copy()
                    effective_unhealed.loc[last_valid_idx + freq_offset:] = False
                    
                    if effective_unhealed.any():
                        gap_groups_post = (effective_unhealed != effective_unhealed.shift()).cumsum()
                        gaps_post = effective_unhealed[effective_unhealed].groupby(gap_groups_post[effective_unhealed])
                        self.audit_report["max_gap_after"] = float(gaps_post.size().max() * freq_min)
                    else:
                        self.audit_report["max_gap_after"] = 0.0
                else:
                    self.audit_report["max_gap_after"] = 1440.0
            else:
                self.audit_report["max_gap_after"] = 0.0
            
            # v4.8.1: Ensure audit report fields are correctly populated for summary
            self.audit_report["total_rows_retained"] = len(final_df) # Will be updated after cleanup
            
            self.audit_report["num_islands_generated"] = current_island_id + 1

            if self.audit_report["max_gap_after"] > 60:
                logger.warning(f"❌ CRITICAL GAP REMAINING: {self.audit_report['max_gap_after']}m in {self.audit_report['file_id']}")

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(f"[transform] Level 1 Healing or reindexing failed: {e}\n{tb}. Falling back to clean dropna.")

        # ── Post-Healing Audit and Cleanup ───────────────────────────────
        # Ensure flow cols have NO NaNs (default to 0.0)
        final_df[actual_flow_cols] = final_df[actual_flow_cols].fillna(0.0)

        # Final cleanup: drop rows that STILL have NaNs (usually just first DS bars)
        # we only expect NaNs now in state variables that couldn't be bfilled (start of day)
        final_df.dropna(inplace=True)
        
        # v4.8.1: Explicit population of rows retained in audit report
        self.audit_report["total_rows_retained"] = int(len(final_df))
        
        logger.info(f"💓 Heartbeat [Post-Cleanup]: {len(final_df)} rows")

        # ── Grouped Feature Engineering (Island Split v4.6) ──────────────────
        final_df = final_df.copy() # Defragment before wide column expansion
        final_df['log_volume'] = np.log1p(final_df['tick_count'])
        
        # Resolve labels
        ds_lbl = str(self._delta_short_min)
        dl_lbl = str(self._delta_long_min)
        ds = self._delta_short
        dl = self._delta_long
        vpin_l = f"{getattr(self, '_vpin_window_min', 25)}"
        vpin_col = f'vpin_min{vpin_l}'

        # Define all expected sniper/institutional columns beforehand
        sniper_institutional_cols = [
            f'ofi_delta_{ds_lbl}', f'ofi_delta_{dl_lbl}',
            f'bid_rdi_delta_{ds_lbl}', f'bid_rdi_delta_{dl_lbl}',
            f'ask_rdi_delta_{ds_lbl}', f'ask_rdi_delta_{dl_lbl}',
            f'micro_price_delta_{ds_lbl}', f'micro_price_delta_{dl_lbl}',
            'book_asymmetry_v5', 'spread_zscore_60', vpin_col,
            'kyle_lambda', 'bid_deep_ratio', 'ask_deep_ratio', 'bid_convexity', 'ask_convexity'
        ]

        def process_island_group(group_df):
            group_df = group_df.copy()
            
            # v4.8.5 Gold: Explicitly capture island_id for restoration
            island_val = group_df['island_id'].iloc[0] if 'island_id' in group_df.columns else 0
            
            # Pre-initialize sniper columns to NaN to ensure they exist regardless of group size
            for col in sniper_institutional_cols:
                if col not in group_df.columns:
                    group_df[col] = np.nan

            if len(group_df) < 2: 
                group_df['island_id'] = island_val
                return group_df
            
            prev_c = group_df['close'].shift(1)
            
            # Candle Shape
            group_df['body'] = np.log(group_df['close'] / group_df['open'])
            group_df['upper_wick'] = (group_df['high'] - np.maximum(group_df['open'], group_df['close'])) / (prev_c + 1e-9)
            group_df['lower_wick'] = (np.minimum(group_df['open'], group_df['close']) - group_df['low']) / (prev_c + 1e-9)
            group_df['log_ret_close'] = np.log(group_df['close'] / (prev_c + 1e-9))
            
            # Sniper Pivot: Multi-Scale Shock Features
            group_df[f'ofi_delta_{ds_lbl}']           = group_df['ofi'].diff(ds)
            group_df[f'bid_rdi_delta_{ds_lbl}']       = group_df['bid_rdi'].diff(ds)
            group_df[f'ask_rdi_delta_{ds_lbl}']       = group_df['ask_rdi'].diff(ds)
            group_df[f'micro_price_delta_{ds_lbl}']   = group_df['close'].pct_change(ds)

            group_df[f'ofi_delta_{dl_lbl}']           = group_df['ofi'].diff(dl)
            group_df[f'bid_rdi_delta_{dl_lbl}']       = group_df['bid_rdi'].diff(dl)
            group_df[f'ask_rdi_delta_{dl_lbl}']       = group_df['ask_rdi'].diff(dl)
            group_df[f'micro_price_delta_{dl_lbl}']   = group_df['close'].pct_change(dl)
            
            # Institutional Features
            n_asym = self._book_asym_depth
            sb_n = sum(group_df[f"bid_{i}_s"] for i in range(n_asym))
            sa_n = sum(group_df[f"ask_{i}_s"] for i in range(n_asym))
            group_df['book_asymmetry_v5'] = np.log((sb_n + 1e-9) / (sa_n + 1e-9))
            
            roll_spread = group_df['max_spread'].rolling(window=self._spread_zscore_window, min_periods=1)
            group_df['spread_zscore_60'] = (group_df['max_spread'] - roll_spread.mean()) / (roll_spread.std() + 1e-9)
            
            group_df[vpin_col] = group_df['ofi'].abs().rolling(self._vpin_window).sum() / (sb_n + sa_n + 1e-9)
            group_df['kyle_lambda'] = group_df[f'micro_price_delta_{ds_lbl}'] / (group_df[f'ofi_delta_{ds_lbl}'].abs() + 1e-9)
            
            dbs = self._deep_book_start
            sb_deep = sum(group_df[f"bid_{i}_s"] for i in range(dbs, self.levels))
            sa_deep = sum(group_df[f"ask_{i}_s"] for i in range(dbs, self.levels))
            group_df['bid_deep_ratio'] = sb_deep / (sb_n + 1e-9)
            group_df['ask_deep_ratio'] = sa_deep / (sa_n + 1e-9)
            
            cn, cf = self._convexity_near_end, self._convexity_far_end
            sb0 = sum(group_df[f"bid_{i}_s"] for i in range(1, cn + 1))
            sb1 = sum(group_df[f"bid_{i}_s"] for i in range(cn + 1, cf + 1))
            sa0 = sum(group_df[f"ask_{i}_s"] for i in range(1, cn + 1))
            sa1 = sum(group_df[f"ask_{i}_s"] for i in range(cn + 1, cf + 1))
            group_df['bid_convexity'] = sb0 / (sb1 + 1e-9)
            group_df['ask_convexity'] = sa0 / (sa1 + 1e-9)
            
            # v4.8.5 Gold: Restore island_id to ensure it's not dropped by groupby().apply()
            group_df['island_id'] = island_val
            
            return group_df

        # Apply transformations isolated by island
        final_df = final_df.groupby('island_id', group_keys=False).apply(process_island_group)

        # Final cleanup for all rolling/diff features (NaNs to 0, Infs to 0)
        # Ensure all columns exist in final_df before cleanup to avoid KeyError/None of Index
        missing_from_final = [c for c in sniper_institutional_cols if c not in final_df.columns]
        if missing_from_final:
            for c in missing_from_final:
                final_df[c] = 0.0

        final_df[sniper_institutional_cols] = final_df[sniper_institutional_cols].replace([np.inf, -np.inf], 0).fillna(0)

        # ── Final Feature List (dynamic delta labels based on real minutes) ──────────
        dynamic_features = [
            'ofi', f'ofi_delta_{ds_lbl}', f'ofi_delta_{dl_lbl}',
            'micro_price_momentum', f'micro_price_delta_{ds_lbl}', f'micro_price_delta_{dl_lbl}',
            'bid_rdi', f'bid_rdi_delta_{ds_lbl}', f'bid_rdi_delta_{dl_lbl}',
            'ask_rdi', f'ask_rdi_delta_{ds_lbl}', f'ask_rdi_delta_{dl_lbl}',
            'spread_zscore_60', vpin_col,
            'kyle_lambda', 'bid_deep_ratio', 'ask_deep_ratio', 'bid_convexity', 'ask_convexity', 'book_asymmetry_v5', 'pressure_ratio'
        ]
        agg_features = [
            'body', 'upper_wick', 'lower_wick', 'log_ret_close',
            'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume', 'tick_count'
        ] + dynamic_features

        # Keep aggregated features + 'close' (label base) + raw orderbook levels
        # Fix: exclude processed sniper/institutional columns from raw ob_cols to prevent duplicates
        ob_cols = [c for c in final_df.columns if (('bid_' in c or 'ask_' in c) 
                   and not any(x in c for x in ['_slope', '_rdi', '_delta_', '_asymmetry', '_convexity']))]
        
        # Build final list and deduplicate while preserving order
        raw_final_cols = agg_features + ['close', 'island_id'] + ob_cols
        final_cols = []
        seen = set()
        for c in raw_final_cols:
            if c in final_df.columns and c not in seen:
                final_cols.append(c)
                seen.add(c)
        
        final_df = final_df[final_cols]

        # Log the new column set for traceability
        new_cols_present = [c for c in dynamic_features if c in final_df.columns]

        # ── Strategic Clipping (v4.5 Patch) ───────────────────────────
        final_df = self._apply_soft_clipping(final_df)

        # ── Final Safety Sweep ───────────────────────────────────────────
        # Replace any residual Infs or NaNs created during feature eng (e.g. log(0))
        final_df.replace([np.inf, -np.inf], 0, inplace=True)
        final_df.fillna(0, inplace=True)

        return final_df

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
        
        ds_lbl = str(self._delta_short_min)
        dl_lbl = str(self._delta_long_min)
        vpin_col = f"vpin_min{self._vpin_window_min}"

        flow_cols = [
            'ofi', f'ofi_delta_{ds_lbl}', f'ofi_delta_{dl_lbl}',
            'micro_price_momentum', f'micro_price_delta_{ds_lbl}', f'micro_price_delta_{dl_lbl}',
            'bid_slope', 'ask_slope',
            'bid_rdi', f'bid_rdi_delta_{ds_lbl}', f'bid_rdi_delta_{dl_lbl}',
            'ask_rdi', f'ask_rdi_delta_{ds_lbl}', f'ask_rdi_delta_{dl_lbl}',
            'book_asymmetry_v5', 'spread_zscore_60', vpin_col,
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
