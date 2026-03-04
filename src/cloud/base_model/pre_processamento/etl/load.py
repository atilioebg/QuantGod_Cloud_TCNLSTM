import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ── Canonical Schema ──────────────────────────────────────────────────────────
# All parquets produced by the ETL must have identical dtypes so downstream
# consumers (run_labelling, split_dataset, training) can concat them safely.
#
#   float32  → all continuous feature columns (saves ~50% vs float64)
#   int32    → island_id         (never > 2^31 islands in a day)
#   int64    → tick_count        (aggregated counts, stay int)
#   object   → kept as-is (string metadata columns, if any)

_FORCED_INT32  = {"island_id"}
_FORCED_INT64  = {"tick_count"}
_KEEP_AS_IS    = {"island_id", "tick_count"}  # skip the float cast for these


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all columns to the canonical ETL schema:
    - float columns → float32  (except _KEEP_AS_IS)
    - island_id     → int32
    - tick_count    → int64 (fillna 0 first so no float coercion from NaN)
    """
    for col in df.columns:
        if col in _FORCED_INT32:
            df[col] = df[col].fillna(0).astype(np.int32)
        elif col in _FORCED_INT64:
            df[col] = df[col].fillna(0).astype(np.int64)
        elif col not in _KEEP_AS_IS and pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif col not in _KEEP_AS_IS and pd.api.types.is_integer_dtype(df[col]):
            # Promote ambiguous int (int8/int16/int32) to consistent int64
            df[col] = df[col].astype(np.int64)
    return df


class DataLoader:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_parquet(self, df: pd.DataFrame, filename: str, compression: str = "snappy") -> bool:
        """Enforces canonical schema and saves DataFrame as Parquet."""
        if df.empty:
            logger.warning(f"DataFrame is empty. Skipping save for {filename}")
            return False

        output_path = self.output_dir / filename
        try:
            df = enforce_schema(df)
            df.to_parquet(output_path, compression=compression, index=True)
            logger.info(f"Successfully saved {len(df)} rows to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving parquet {output_path}: {e}")
            return False


if __name__ == "__main__":
    # Test
    loader = DataLoader("./test_output")
    # loader.save_parquet(pd.DataFrame({"a": [1, 2]}), "test.parquet")
