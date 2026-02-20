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
            logger.warning(f"Columns with NaNs: {nan_cols}")
        else:
            logger.info("No NaNs found.")

        # 2. Check for Infs
        infs = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if infs > 0:
            logger.warning(f"Found {infs} Infinite values in {name}")
        else:
            logger.info("No Infinite values found.")

        # 3. Check for Chronological Order
        if not df.index.is_monotonic_increasing:
            logger.error("Dataset is NOT chronologically sorted!")
        else:
            logger.info("Dataset is correctly sorted.")

        # 4. Check for Time Gaps (assuming 1min resampled)
        diffs = df.index.to_series().diff().dropna()
        max_gap = diffs.max()
        if max_gap > pd.Timedelta(minutes=5):
            logger.warning(f"Found large time gap: {max_gap}")
        
        logger.info(f"Validation complete for {name}. Total rows: {len(df)}")
        return nans == 0 and infs == 0
