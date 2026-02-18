import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_parquet(self, df: pd.DataFrame, filename: str, compression: str = "snappy"):
        """Saves DataFrame as Parquet with given compression."""
        if df.empty:
            logger.warning(f"DataFrame is empty. Skipping save for {filename}")
            return
        
        output_path = self.output_dir / filename
        try:
            df.to_parquet(output_path, compression=compression, index=True)
            logger.info(f"Successfully saved {len(df)} rows to {output_path}")
        except Exception as e:
            logger.error(f"Error saving parquet {output_path}: {e}")

if __name__ == "__main__":
    # Test
    loader = DataLoader("./test_output")
    # loader.save_parquet(pd.DataFrame({"a": [1, 2]}), "test.parquet")
