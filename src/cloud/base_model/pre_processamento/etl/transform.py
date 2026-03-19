import json
import polars as pl
import numpy as np
import logging
import yaml
from typing import Dict, List, Optional, Any
import pickle
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

# ── ETL config loader ───────────────────────────────────────────
def _parse_resample_minutes(freq: str) -> int:
    """
    Converts a resample frequency string (e.g. '1min', '5min') to integer minutes.
    """
    freq = freq.strip()
    for suffix in ('min', 'T', 'Min'):
        if freq.endswith(suffix):
            return max(1, int(freq.replace(suffix, '')))
    if freq.endswith(('h', 'H')):
        return int(freq[:-1]) * 60
    return 1


def _to_polars_freq(freq: str) -> str:
    """
    Converts a Pandas-style frequency alias to a Polars duration string.

    Polars uses 'm' for minutes (NOT 'min').
    Mapping:
        '5min' → '5m'   '1T' → '1m'   '1h' → '1h'
        '1H'   → '1h'   '1d' → '1d'

    Used anywhere a Polars API receives a duration/interval string
    (group_by_dynamic, datetime_range, etc.).
    """
    freq = freq.strip()
    # Pandas minute aliases → Polars 'm'
    for suffix in ('min', 'T', 'Min'):
        if freq.endswith(suffix):
            n = freq.replace(suffix, '') or '1'
            return f"{n}m"
    # Hour aliases
    if freq.endswith('H'):
        n = freq[:-1] or '1'
        return f"{n}h"
    # Already Polars-style or day/week/etc — return as-is
    return freq


def _load_etl_config() -> dict:
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

        resolved["clipping"] = etl.get("clipping", {"enabled": False})
        resolved["audit"]    = etl.get("audit", {"generate_report": False})
        return resolved
    except Exception:
        resample_min = _parse_resample_minutes(_DEFAULTS["resample_freq"])
        fallback = dict(_DEFAULTS)
        fallback["resample_min"] = resample_min
        for min_key, default_min in _DEFAULT_MINS.items():
            bar_key           = min_key.replace("_min", "")
            fallback[bar_key] = max(1, default_min // resample_min)
            fallback[min_key] = default_min
        fallback["clipping"] = {"enabled": False}
        return fallback


_ETL_CFG = _load_etl_config()
_FLOW_DEPTH = _ETL_CFG["flow_depth"]


class L2Transformer:
    def __init__(self, levels: int = 200, sampling_ms: int = 1000,
                 flow_depth: int = None, etl_cfg: dict = None):
        cfg = dict(_ETL_CFG)
        if etl_cfg:
            cfg.update(etl_cfg)

        self.levels = levels
        self.sampling_ms = sampling_ms
        self.flow_depth  = flow_depth if flow_depth is not None else cfg["flow_depth"]

        self._resample_freq        = cfg["resample_freq"]
        self._resample_min         = int(cfg["resample_min"])
        self._spread_zscore_window = int(cfg["spread_zscore_window"])
        self._vpin_window          = int(cfg["vpin_window"])
        self._vpin_window_min      = int(cfg.get("vpin_window_min", 25))
        self._delta_short          = int(cfg["delta_short"])
        self._delta_long           = int(cfg["delta_long"])
        self._delta_short_min      = int(cfg["delta_short_min"])
        self._delta_long_min       = int(cfg["delta_long_min"])
        self._book_asym_depth      = int(cfg["book_asymmetry_depth"])
        self._deep_book_start      = int(cfg["deep_book_start"])
        self._convexity_near_end   = int(cfg["convexity_near_end"])
        self._convexity_far_end    = int(cfg["convexity_far_end"])

        self._clipping_cfg = cfg.get("clipping", {"enabled": False})
        self._etl_cfg = cfg
        self.audit_report = {
            "file_id": "unknown",
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

        self.bids_book: Dict[float, float] = {}
        self.asks_book: Dict[float, float] = {}
        self.last_sample_ts: int = -1
        self._prev_bids: list = []
        self._prev_asks: list = []
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
        msg_type = msg.get("type")
        ts = msg.get("ts")
        data = msg.get("data", {})

        if not ts: return None

        if msg_type == "snapshot":
            self.bids_book = {float(p): float(s) for p, s in data.get("b", [])}
            self.asks_book = {float(p): float(s) for p, s in data.get("a", [])}
        else:
            for p, s in data.get("b", []):
                price, size = float(p), float(s)
                if size == 0: self.bids_book.pop(price, None)
                else: self.bids_book[price] = size
            for p, s in data.get("a", []):
                price, size = float(p), float(s)
                if size == 0: self.asks_book.pop(price, None)
                else: self.asks_book[price] = size

        if self.last_sample_ts == -1 or ts - self.last_sample_ts >= self.sampling_ms:
            if self.last_sample_ts != -1 and ts - self.last_sample_ts > 2 * self.sampling_ms:
                logger.debug(f"[transform] Timestamp gap ({ts - self.last_sample_ts}ms). Resetting T-1 state.")
                self._prev_bids = []
                self._prev_asks = []
                self._prev_micro_price = np.nan
                self._prev_obi_l0 = 0.0
            self.last_sample_ts = (ts // self.sampling_ms) * self.sampling_ms
            return self._capture_state(self.last_sample_ts)

        return None

    def _capture_state(self, ts: int) -> Dict:
        sorted_bids = sorted(self.bids_book.keys(), reverse=True)[:self.levels]
        sorted_asks = sorted(self.asks_book.keys())[:self.levels]

        if sorted_bids and sorted_asks:
            if sorted_bids[0] >= sorted_asks[0]:
                logger.warning(f"Orderbook crossed at {ts}: Bid {sorted_bids[0]} >= Ask {sorted_asks[0]}")

        row = {"ts": ts}

        bid0_p = sorted_bids[0] if sorted_bids else np.nan
        bid0_s = self.bids_book[bid0_p] if sorted_bids else 0.0
        ask0_p = sorted_asks[0] if sorted_asks else np.nan
        ask0_s = self.asks_book[ask0_p] if sorted_asks else 0.0
        total_0 = bid0_s + ask0_s

        if sorted_bids and sorted_asks:
            micro_price = (bid0_p * ask0_s + ask0_p * bid0_s) / total_0 if total_0 > 0 else np.nan
            obi_l0 = (bid0_s - ask0_s) / total_0 if total_0 > 0 else 0.0
            row['spread'] = ask0_p - bid0_p
        else:
            micro_price = np.nan
            obi_l0 = 0.0
            row['spread'] = np.nan

        row['micro_price'] = micro_price
        row['obi_l0'] = obi_l0

        n = self.flow_depth
        top_bids = [(p, self.bids_book[p]) for p in sorted_bids[:n]]
        top_asks = [(p, self.asks_book[p]) for p in sorted_asks[:n]]

        bid_vols = [s for _, s in top_bids]
        ask_vols = [s for _, s in top_asks]
        bid_vol_n = sum(bid_vols)
        ask_vol_n = sum(ask_vols)

        row['deep_obi_5'] = (bid_vol_n - ask_vol_n) / (bid_vol_n + ask_vol_n) if (bid_vol_n + ask_vol_n) > 0 else 0.0

        if len(top_bids) >= n and top_bids[0][0] != top_bids[-1][0]:
            row['bid_slope'] = bid_vol_n / abs(top_bids[0][0] - top_bids[-1][0])
        else:
            row['bid_slope'] = 0.0

        if len(top_asks) >= n and top_asks[0][0] != top_asks[-1][0]:
            row['ask_slope'] = ask_vol_n / abs(top_asks[-1][0] - top_asks[0][0])
        else:
            row['ask_slope'] = 0.0

        bid_depth_1_4 = sum(bid_vols[1:]) if len(bid_vols) > 1 else 0.0
        ask_depth_1_4 = sum(ask_vols[1:]) if len(ask_vols) > 1 else 0.0
        row['bid_rdi'] = (bid_depth_1_4 / bid0_s) if bid0_s > 0 else 0.0
        row['ask_rdi'] = (ask_depth_1_4 / ask0_s) if ask0_s > 0 else 0.0

        prev_b = self._prev_bids
        prev_a = self._prev_asks

        if prev_b and prev_a:
            ofi_total = 0.0
            for i in range(n):
                if i < len(top_bids) and i < len(prev_b):
                    cp, cs = top_bids[i]
                    pp, ps = prev_b[i]
                    if cp > pp:    ofi_bid_i = cs
                    elif cp == pp: ofi_bid_i = cs - ps
                    else:          ofi_bid_i = -ps
                else:
                    ofi_bid_i = 0.0

                if i < len(top_asks) and i < len(prev_a):
                    cp, cs = top_asks[i]
                    pp, ps = prev_a[i]
                    if cp < pp:    ofi_ask_i = cs
                    elif cp == pp: ofi_ask_i = cs - ps
                    else:          ofi_ask_i = -ps
                else:
                    ofi_ask_i = 0.0

                ofi_total += ofi_bid_i - ofi_ask_i

            row['ofi'] = ofi_total

            if not np.isnan(micro_price) and not np.isnan(self._prev_micro_price) and self._prev_micro_price > 0:
                row['micro_price_momentum'] = np.log(micro_price / self._prev_micro_price)
            else:
                row['micro_price_momentum'] = 0.0

            prev_obi = self._prev_obi_l0
            if abs(prev_obi) > 1e-9:
                row['pressure_ratio'] = (obi_l0 - prev_obi) / abs(prev_obi)
            else:
                row['pressure_ratio'] = 0.0
        else:
            row['ofi'] = 0.0
            row['micro_price_momentum'] = 0.0
            row['pressure_ratio'] = 0.0

        self._prev_bids = top_bids
        self._prev_asks = top_asks
        self._prev_micro_price = micro_price
        self._prev_obi_l0 = obi_l0

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

    # ─────────────────────────────────────────────────────────────────────────
    # Polars Feature Engineering Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def apply_feature_engineering(self, rows_or_df) -> pl.DataFrame:
        """
        [v6.0 Event-Driven] Unified wrapper calling EventSampler.
        """
        # ── 0. Build Polars DataFrame ────────────────────────────────────────
        if isinstance(rows_or_df, dict):
            df = pl.DataFrame(rows_or_df)
        elif isinstance(rows_or_df, pl.DataFrame):
            df = rows_or_df
        else:
            import pandas as pd
            if isinstance(rows_or_df, pd.DataFrame):
                df = pl.from_pandas(rows_or_df.reset_index(drop=True))
            else:
                logger.warning("apply_feature_engineering: unknown input type, skipping")
                return pl.DataFrame()

        if df.is_empty():
            logger.warning("apply_feature_engineering received an empty DataFrame")
            return df

        if "ts" in df.columns and "datetime" not in df.columns:
            df = df.with_columns(pl.from_epoch("ts", time_unit="ms").alias("datetime"))
            
        # Initialize EventSampler and process
        from .event_sampler import EventSampler
        sampler = EventSampler(self._etl_cfg)
        
        # 1. Compute Base Event Bars (Dollar, Tick, Info, Time)
        df_bars = sampler.compute_event_bars(df)
        if df_bars.is_empty():
            return df_bars
            
        # 2. Derive Institutional Features & Rollings
        df_feats = sampler.apply_feature_engineering_bars(df_bars)
        
        # Sincronizar relatório de auditoria do sampler para o transformer
        self.audit_report.update(sampler.audit_report)
        
        # 3. Soft Clipping (from transform.py local method)
        df_final = self._apply_soft_clipping_pl(df_feats)
        
        # 4. Cast to canonical schema to avoid downstream breaks
        cast_exprs = []
        for c in df_final.columns:
            if c == "island_id":
                cast_exprs.append(pl.col(c).cast(pl.Int32))
            elif c == "tick_count":
                cast_exprs.append(pl.col(c).cast(pl.Int64))
            elif df_final.schema[c] in (pl.Float32, pl.Float64):
                cast_exprs.append(pl.col(c).cast(pl.Float32))
        if cast_exprs:
            df_final = df_final.with_columns(cast_exprs)
            
        return df_final

    def _apply_soft_clipping_pl(self, df: pl.DataFrame) -> pl.DataFrame:
        """Polars implementation of Winsorization (clipping based on P99 * multiplier)."""
        if not self._clipping_cfg.get("enabled", False):
            return df

        target_cols = self._clipping_cfg.get("target_columns", [])
        multiplier  = self._clipping_cfg.get("p99_multiplier", 10)
        noise_floor = self._clipping_cfg.get("noise_floor", 1e-6)

        total_cells = len(df) * len(target_cols) if target_cols else 1
        clipped_count = 0

        for col in target_cols:
            if col not in df.columns:
                continue

            p99 = df[col].quantile(0.99)

            if p99 is None or p99 < noise_floor:
                p50 = df[col].median()
                if p50 is None or p50 <= 0.0:
                    continue
                max_val = df[col].max()
                fallback_threshold = max(p50 * 100, noise_floor * 1000)
                if max_val <= fallback_threshold:
                    continue
                count = (df[col] > fallback_threshold).sum()
                clipped_count += count
                self.audit_report["clipping_events"][col] = {
                    "count": int(count), "max_original": float(max_val),
                    "threshold": float(fallback_threshold), "fallback": "100x_median"
                }
                logger.warning(
                    f"☢️ [CLIPPING/BIMODAL] {self.audit_report['file_id']}: {col} "
                    f"clipped at 100x P50 ({fallback_threshold:.4e})"
                )
                df = df.with_columns(
                    pl.col(col).clip(upper_bound=fallback_threshold).alias(col)
                )
                continue

            threshold = p99 * multiplier
            if (df[col] > threshold).any():
                max_val = df[col].max()
                count   = (df[col] > threshold).sum()
                clipped_count += count
                self.audit_report["clipping_events"][col] = {
                    "count": int(count), "max_original": float(max_val),
                    "threshold": float(threshold)
                }
                logger.warning(
                    f"☢️ [CLIPPING] {self.audit_report['file_id']}: {col} "
                    f"max ({max_val:.4f}) → {threshold:.4f} (10x P99)"
                )
                df = df.with_columns(
                    pl.col(col).clip(upper_bound=threshold).alias(col)
                )

        self.audit_report["outlier_density"] = (clipped_count / total_cells) * 100 if total_cells > 0 else 0.0
        self.audit_report["clipped_count"]   = int(clipped_count)
        return df

    def apply_zscore(self, df: pl.DataFrame, scaler_or_path: Optional[Any] = None) -> pl.DataFrame:
        """
        Normalizes features. Returns a pl.DataFrame.
        - scaler_or_path: Path to .pkl, or a pre-loaded scaler object/dict.
        """
        import pandas as pd
        import joblib
        from sklearn.preprocessing import StandardScaler, RobustScaler

        ds_lbl   = str(self._delta_short_min)
        dl_lbl   = str(self._delta_long_min)
        vpin_col = f"vpin_min{self._vpin_window_min}"

        original_cols = [c for c in [
            'body', 'upper_wick', 'lower_wick',
            'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi',
            'log_volume', 'log_ret_close',
        ] if c in df.columns]

        flow_cols = [c for c in [
            'ofi', f'ofi_delta_{ds_lbl}', f'ofi_delta_{dl_lbl}',
            'micro_price_momentum', f'micro_price_delta_{ds_lbl}', f'micro_price_delta_{dl_lbl}',
            'bid_rdi', f'bid_rdi_delta_{ds_lbl}', f'bid_rdi_delta_{dl_lbl}',
            'ask_rdi', f'ask_rdi_delta_{ds_lbl}', f'ask_rdi_delta_{dl_lbl}',
            'book_asymmetry_v5', 'spread_zscore_60', vpin_col,
            'kyle_lambda', 'bid_deep_ratio', 'ask_deep_ratio',
            'bid_convexity', 'ask_convexity', 'pressure_ratio',
            'mean_bid_slope', 'mean_ask_slope', 'mean_spread',
        ] if c in df.columns]

        if df.is_empty():
            return df

        # Convert to pandas for sklearn
        import gc
        pdf = df.to_pandas()
        
        # v5.1: Clear Polars memory if we have a large dataframe
        # (df is still in scope, but we can't delete it safely as it's an arg)
        gc.collect()

        scaler_bundle = None
        if scaler_or_path is not None:
            if isinstance(scaler_or_path, (str, Path)):
                # Ensure we have an absolute path if it's relative
                path = Path(scaler_or_path)
                if not path.is_absolute():
                     # Try relative to project root (5 levels up from this file)
                     project_root = Path(__file__).parents[5]
                     path = project_root / path
                
                if path.exists():
                    try:
                        scaler_bundle = joblib.load(path)
                    except:
                        with open(path, 'rb') as f:
                            scaler_bundle = pickle.load(f)
            else:
                scaler_bundle = scaler_or_path

        if scaler_bundle is not None:
            # Logic for bundled (dict) or single scaler
            if isinstance(scaler_bundle, dict) and 'standard' in scaler_bundle:
                # Legacy multi-scaler support
                std_sc = scaler_bundle['standard']
                rob_sc = scaler_bundle['robust']
                # Try to use feature names from scaler if available
                if hasattr(std_sc, 'feature_names_in_'):
                    cols = std_sc.feature_names_in_
                    pdf[cols] = std_sc.transform(pdf[cols])
                if hasattr(rob_sc, 'feature_names_in_'):
                    cols = rob_sc.feature_names_in_
                    pdf[cols] = rob_sc.transform(pdf[cols])
            else:
                 # Single scaler for everything (Standard behavior for QuantGod v4+)
                 # CRITICAL: We MUST use the columns in the order the scaler expects.
                 if hasattr(scaler_bundle, 'feature_names_in_'):
                     target_cols = [c for c in scaler_bundle.feature_names_in_ if c in pdf.columns]
                     if len(target_cols) != len(scaler_bundle.feature_names_in_):
                         missing = set(scaler_bundle.feature_names_in_) - set(pdf.columns)
                         logger.warning(f"⚠️ Scaler expects {len(scaler_bundle.feature_names_in_)} features, but {len(target_cols)} found. Missing: {missing}")
                     
                     # Transform using the exact order of the scaler, passing .values to suppress sklearn warnings
                     pdf[scaler_bundle.feature_names_in_] = scaler_bundle.transform(pdf[scaler_bundle.feature_names_in_].values)
                 else:
                     # Fallback to current columns if scaler has no names (not ideal)
                     target_cols = [c for c in pdf.columns if c not in ['datetime', 'island_id', 'open', 'high', 'low', 'close']]
                     pdf[target_cols] = scaler_bundle.transform(pdf[target_cols].values)
        else:
            # Fallback: fit a new one (not recommended for production inference)
            logger.warning("⚠️ No scaler provided to apply_zscore. Fitting a NEW one (Inference will be WRONG!)")
            std_sc = StandardScaler()
            rob_sc = RobustScaler()
            if original_cols: pdf[original_cols] = std_sc.fit_transform(pdf[original_cols])
            if flow_cols:     pdf[flow_cols]     = rob_sc.fit_transform(pdf[flow_cols])

        # v5.1: Convert back to Polars and clean up the bulky Pandas copy
        res_df = pl.from_pandas(pdf)
        del pdf
        gc.collect()
        return res_df
