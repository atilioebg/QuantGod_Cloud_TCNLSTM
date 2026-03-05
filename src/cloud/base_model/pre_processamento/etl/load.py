import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_parquet(self, df, filename: str, compression: str = "snappy") -> bool:
        """
        Saves a DataFrame (Polars or Pandas) as Parquet with canonical schema.

        - pl.DataFrame → write_parquet directly (schema already enforced by transform.py)
        - pd.DataFrame → enforce_schema() then to_parquet (legacy fallback)
        """
        if df is None:
            logger.warning(f"DataFrame is None. Skipping save for {filename}")
            return False

        output_path = self.output_dir / filename

        try:
            if isinstance(df, pl.DataFrame):
                if df.is_empty():
                    logger.warning(f"DataFrame is empty. Skipping save for {filename}")
                    return False
                df.write_parquet(output_path, compression=compression)
                logger.info(f"Successfully saved {len(df)} rows to {output_path}")
                return True
            else:
                # Pandas fallback
                if df.empty:
                    logger.warning(f"DataFrame is empty. Skipping save for {filename}")
                    return False
                df = _enforce_pandas_schema(df)
                df.to_parquet(output_path, compression=compression, index=True)
                logger.info(f"Successfully saved {len(df)} rows to {output_path}")
                return True
        except Exception as e:
            logger.error(f"Error saving parquet {output_path}: {e}")
            return False


# ── Pandas schema enforcement (legacy fallback only) ──────────────────────────
_FORCED_INT32 = {"island_id"}
_FORCED_INT64 = {"tick_count"}
_KEEP_AS_IS   = {"island_id", "tick_count"}


def _enforce_pandas_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast Pandas DataFrame to the canonical schema (used only for legacy Pandas path)."""
    for col in df.columns:
        if col in _FORCED_INT32:
            df[col] = df[col].fillna(0).astype(np.int32)
        elif col in _FORCED_INT64:
            df[col] = df[col].fillna(0).astype(np.int64)
        elif col not in _KEEP_AS_IS and pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif col not in _KEEP_AS_IS and pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(np.int64)
    return df


if __name__ == "__main__":
    loader = DataLoader("./test_output")

