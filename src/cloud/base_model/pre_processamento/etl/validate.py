import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate_integrity(df: pd.DataFrame, name: str = "Dataset"):
        """Performs basic integrity checks on the processed data."""
        logger.info(f"--- Validating {name} ---")
        
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
        # Validate if 5min features (delta_5) match the rolling window of 1min features
        if 'ofi' in df.columns and 'ofi_delta_5' in df.columns:
            # Reconstruct what ofi_delta_5 should be: ofi(t) - ofi(t-5) -> this is ofi.diff(5)
            # transform.py uses: final_df['ofi_delta_5'] = final_df['ofi'].diff(5)
            # We verify this in-situ
            reconstructed_delta = df['ofi'].diff(5).fillna(0)
            diff_check = (df['ofi_delta_5'].fillna(0) - reconstructed_delta).abs().max()
            if diff_check > 1e-7:
                 logger.warning(f"⚠️ CROSS-SCALE INCONSISTENCY: ofi_delta_5 drift detected ({diff_check})")
            else:
                 logger.info("Cross-scale consistency verified (OFI).")

        # 6. Distribution Sanity (Outlier Destruction Prevention)
        ratio_features = ['kyle_lambda', 'vpin_lite_5', 'bid_deep_ratio', 'ask_deep_ratio']
        for feat in ratio_features:
            if feat in df.columns:
                p99 = df[feat].quantile(0.99)
                max_val = df[feat].max()
                if p99 > 0 and max_val > 50 * p99:
                    logger.warning(f"☢️ CRITICAL OUTLIER: {feat} max ({max_val:.2f}) is > 50x P99 ({p99:.2f}). Potential destructive signal.")

        # 7. Check for Time Gaps (assuming 1min resampled)
        diffs = df.index.to_series().diff().dropna()
        max_gap = diffs.max()
        if max_gap > pd.Timedelta(minutes=5):
            logger.warning(f"Found large time gap: {max_gap}")
        
        logger.info(f"Validation complete for {name}. Total rows: {len(df)}")
        return nans == 0 and infs == 0
