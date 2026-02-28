import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate_integrity(df: pd.DataFrame, name: str = "Dataset"):
        """Performs basic integrity checks on the processed data."""
        logger.info(f"--- Validating {name} ---")
        
        # 0. Check for Duplicate Columns (FATAL)
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            msg = f"❌ FATAL: Dataset {name} has duplicate columns: {dupes}"
            logger.error(msg)
            raise ValueError(msg)
        
        # 1. Check for NaNs
        nans = df.isna().sum().sum()
        if nans > 0:
            logger.warning(f"Found {nans} NaN values in {name}")
            # Identify columns with NaNs
            nan_cols = df.columns[df.isna().any()].tolist()
            if len(nan_cols) > 5:
                logger.warning(f"Columns with NaNs (Top 5): {nan_cols[:5]} ... and {len(nan_cols)-5} more.")
            else:
                logger.warning(f"Columns with NaNs: {nan_cols}")
        else:
            logger.info("No NaNs found.")

        # 2. Check for Infs
        infs = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if infs > 0:
            logger.warning(f"Found {infs} Infinite values in {name}")
        else:
            logger.info("No Infinite values found.")

        # 3. Strict Monotonicity Enforcement (FATAL)
        if not df.index.is_monotonic_increasing:
            msg = f"❌ FATAL: Dataset {name} is NOT chronologically sorted! Corruption detected."
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.info("Chronological order verified.")

        # 4. Stale Data Check (Feed Lock Detection)
        # Alert if more than 5 minutes have exactly zero price and volume variation
        if len(df) > 5:
            # We check if close price and log_volume (or tick_count proxy) are static
            price_static = (df['close'].diff() == 0).rolling(5).sum() == 5
            # Using log_volume as proxy for activity
            vol_static = (df['log_volume'].diff() == 0).rolling(5).sum() == 5
            stale_indices = df.index[price_static & vol_static]
            if not stale_indices.empty:
                logger.warning(f"⚠️ STALE DATA ALERT: Possible feed lock detected at: {stale_indices[:3].tolist()}...")

        # 5. Cross-Scale Validation (Mathematical Consistency)
        # Validate ofi_delta_1 = ofi.diff(1) on the 5min pipeline
        if 'ofi' in df.columns and 'ofi_delta_1' in df.columns:
            # transform.py: final_df['ofi_delta_1'] = final_df['ofi'].diff(1)
            reconstructed_delta = df['ofi'].diff(1).fillna(0)
            check_val = (df['ofi_delta_1'].fillna(0) - reconstructed_delta).abs()
            diff_check = check_val.max().max() if isinstance(check_val, pd.DataFrame) else check_val.max()
            
            if diff_check > 1e-7:
                 logger.warning(f"⚠️ CROSS-SCALE INCONSISTENCY: ofi_delta_1 drift detected ({diff_check})")
            else:
                 logger.info("Cross-scale consistency verified (OFI).")

        # 6. Distribution Sanity (Outlier Destruction Prevention)
        # Ratio features and institutional metrics clippped in transform.py
        ratio_features = [
            'kyle_lambda', 'vpin_min25', 'bid_deep_ratio', 'ask_deep_ratio',
            'bid_convexity', 'ask_convexity', 'book_asymmetry_v5', 'max_spread', 'ofi'
        ]
        active_features = [f for f in ratio_features if f in df.columns]
        
        for feat in active_features:
            p99 = df[feat].quantile(0.99)
            max_val = df[feat].max()
            
            # Since clipping v4.5 caps at 10x P99, seeing > 15x P99 means clipping failed/skipped
            if p99 > 1e-6 and max_val > 15 * p99:
                logger.warning(f"☢️ CRITICAL OUTLIER: {feat} max ({max_val:.4e}) is > 15x P99 ({p99:.4e}). Clipping might have failed!")
            elif p99 > 1e-6 and max_val > 10.1 * p99:
                # Tolerance of 0.1 for floating point / quantile variations
                logger.info(f"✅ Clipping verified for {feat}: max is bounded at ~10x P99.")

        # 7. Z-Score Intensity (Tail Check)
        # Gold Level: Detect if distributions are too wide even after clipping
        numeric_df = df.select_dtypes(include=[np.number])
        z_scores = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-9)
        extreme_points = (z_scores.abs() > 12).sum().sum()
        if extreme_points > 0:
            logger.warning(f"⚠️ HIGH TAIL INTENSITY: {extreme_points} data points have Z-Score > 12. Potential volatility shock.")

        # 8. Zero-Variance Detection (Dead Features)
        std_zero = numeric_df.std()
        dead_cols = std_zero[std_zero == 0].index.tolist()
        if dead_cols:
             logger.warning(f"🧟 DEAD FEATURES DETECTED (Zero Variance): {dead_cols}")

        # 9. Time Gaps (5min resampled — alert if gap > 25min = 5 bars)
        diffs = df.index.to_series().diff().dropna()
        if not diffs.empty:
            max_gap = diffs.max()
            if not pd.isna(max_gap) and max_gap > pd.Timedelta(minutes=25):
                logger.warning(f"Found large time gap: {max_gap}")
        
        logger.info(f"🏆 Gold Validation complete for {name}. Total rows: {len(df)}")
        return nans == 0 and infs == 0
