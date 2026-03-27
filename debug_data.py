import zipfile
import json
import polars as pl
from pathlib import Path

zip_path = Path("data/backtest/L2/raw_zip/2026-03-17_BTCUSDT_ob200.data.zip")
trade_path = Path("data/backtest/L2/raw_trades/BTCUSDT2026-03-17.csv.gz")

print(f"Checking ZIP: {zip_path}")
try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        print(f"Files in ZIP: {len(z.namelist())}")
        with z.open(z.namelist()[0]) as f:
            first_line = f.readline()
            print(f"First line: {first_line[:100]}...")
except Exception as e:
    print(f"ZIP ERROR: {e}")

print(f"\nChecking TRADE: {trade_path}")
try:
    df = pl.read_csv(trade_path, n_rows=10)
    print(f"Trade data:\n{df.head()}")
except Exception as e:
    print(f"TRADE ERROR: {e}")
