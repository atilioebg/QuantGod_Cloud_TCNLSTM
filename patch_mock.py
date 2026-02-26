import polars as pl
import numpy as np
from pathlib import Path

def patch_mock_data():
    base_dir = Path("data/L2/splits_labelled_SELL_0003_BUY_0003_15min")
    for split in ['train', 'val']:
        mock_file = base_dir / split / f"{split}_mock.parquet"
        if mock_file.exists():
            print(f"Patching {mock_file}...")
            df = pl.read_parquet(mock_file)
            n = len(df)
            
            # Adicionar close, high, low
            if 'close' not in df.columns:
                df = df.with_columns(pl.Series("close", np.random.uniform(50000, 60000, n)))
            if 'high' not in df.columns:
                df = df.with_columns(pl.Series("high", df["close"] * (1 + np.random.uniform(0.001, 0.005, n))))
            if 'low' not in df.columns:
                df = df.with_columns(pl.Series("low", df["close"] * (1 - np.random.uniform(0.001, 0.005, n))))
                
            # Log Volume
            if 'log_volume' not in df.columns:
                df = df.with_columns(pl.Series("log_volume", np.random.uniform(1, 10, n)))
                
            # Adicionar 200 colunas de BID_S e ASK_S
            for i in range(200):
                if f"bid_{i}_s" not in df.columns:
                    df = df.with_columns(pl.Series(f"bid_{i}_s", np.random.uniform(0.1, 5.0, n)))
                if f"ask_{i}_s" not in df.columns:
                    df = df.with_columns(pl.Series(f"ask_{i}_s", np.random.uniform(0.1, 5.0, n)))
                    
            df.write_parquet(mock_file)
            print(f"Patched {split} with new columns.")

if __name__ == "__main__":
    patch_mock_data()
