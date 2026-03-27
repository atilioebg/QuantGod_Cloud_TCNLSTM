import zipfile
from pathlib import Path

zip_path = Path("data/backtest/L2/raw_zip/2026-03-17_BTCUSDT_ob200.data.zip")

count = 0
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(z.namelist()[0]) as f:
        for line in f:
            count += 1
            if count % 1000000 == 0:
                print(f"{count // 1000000}M messages...")
print(f"Total: {count}")
