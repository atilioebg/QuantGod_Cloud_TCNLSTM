import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate_integrity(df: pd.DataFrame, name: str = "Dataset") -> dict:
        """
        Performs basic integrity checks and returns a structured quality report.
        Returns:
            dict: {
                'is_valid': bool,
                'nan_count': int,
                'inf_count': int,
                'dead_features': list,
                'high_tail_count': int,
                'stale_data_detected': bool,
                'max_gap_minutes': float
            }
        """
        logger.info(f"--- Validating {name} ---")
        
        report = {
            'is_valid': True,
            'nan_count': 0,
            'inf_count': 0,
            'dead_features': [],
            'high_tail_count': 0,
            'stale_data_detected': False,
            'max_gap_minutes': 0.0
        }

        # 0. Check for Duplicate Columns (FATAL)
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            msg = f"❌ FATAL: Dataset {name} has duplicate columns: {dupes}"
            logger.error(msg)
            raise ValueError(msg)
        
        # 1. Check for NaNs
        report['nan_count'] = int(df.isna().sum().sum())
        if report['nan_count'] > 0:
            logger.warning(f"Found {report['nan_count']} NaN values in {name}")
            report['is_valid'] = False
        else:
            logger.info("No NaNs found.")

        # 2. Check for Infs
        report['inf_count'] = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
        if report['inf_count'] > 0:
            logger.warning(f"Found {report['inf_count']} Infinite values in {name}")
            report['is_valid'] = False
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
        if len(df) > 5:
            price_static = (df['close'].diff() == 0).rolling(5).sum() == 5
            vol_static = (df['log_volume'].diff() == 0).rolling(5).sum() == 5
            stale_indices = df.index[price_static & vol_static]
            if not stale_indices.empty:
                logger.warning(f"⚠️ STALE DATA ALERT: Possible feed lock detected in {name}")
                report['stale_data_detected'] = True

        # 5. Cross-Scale Validation (Mathematical Consistency)
        if 'ofi' in df.columns and 'ofi_delta_1' in df.columns:
            reconstructed_delta = df['ofi'].diff(1).fillna(0)
            check_val = (df['ofi_delta_1'].fillna(0) - reconstructed_delta).abs()
            diff_check = float(check_val.max().max() if isinstance(check_val, pd.DataFrame) else check_val.max())
            if diff_check > 1e-7:
                 logger.warning(f"⚠️ CROSS-SCALE INCONSISTENCY: ofi_delta_1 drift detected ({diff_check})")
            else:
                 logger.info("Cross-scale consistency verified (OFI).")

        # 6. Distribution Sanity (Outlier Destruction Prevention)
        ratio_features = [
            'kyle_lambda', 'vpin_min25', 'bid_deep_ratio', 'ask_deep_ratio',
            'bid_convexity', 'ask_convexity', 'book_asymmetry_v5', 'max_spread', 'ofi'
        ]
        active_features = [f for f in ratio_features if f in df.columns]
        for feat in active_features:
            p99 = df[feat].quantile(0.99)
            max_val = df[feat].max()
            if p99 > 1e-6 and max_val > 15 * p99:
                logger.warning(f"☢️ CRITICAL OUTLIER: {feat} max ({max_val:.4e}) is > 15x P99. Clipping might have failed!")
            elif p99 > 1e-6 and max_val > 10.1 * p99:
                logger.info(f"✅ Clipping verified for {feat}.")

        # 7. Z-Score Intensity (Tail Check)
        numeric_df = df.select_dtypes(include=[np.number])
        z_scores = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-9)
        report['high_tail_count'] = int((z_scores.abs() > 12).sum().sum())
        if report['high_tail_count'] > 0:
            logger.warning(f"⚠️ HIGH TAIL INTENSITY: {report['high_tail_count']} points with Z-Score > 12.")

        # 8. Zero-Variance Detection (Dead Features)
        std_zero = numeric_df.std()
        report['dead_features'] = std_zero[std_zero == 0].index.tolist()
        if report['dead_features']:
             logger.warning(f"🧟 DEAD FEATURES DETECTED (Zero Variance): {report['dead_features']}")

        # 9. Time Gaps
        diffs = df.index.to_series().diff().dropna()
        if not diffs.empty:
            max_gap = diffs.max()
            report['max_gap_minutes'] = float(max_gap.total_seconds() / 60)
            if max_gap > pd.Timedelta(minutes=25):
                logger.warning(f"Found large time gap: {max_gap}")
        
        logger.info(f"🏆 Gold Validation complete for {name}. Status: {'VALID' if report['is_valid'] else 'INVALID'}")
        return report
