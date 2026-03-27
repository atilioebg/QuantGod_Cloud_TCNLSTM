import zipfile
import os
from pathlib import Path
from tqdm import tqdm

raw_zip_dir = Path("data/backtest/L2/raw_zip")
unzipped_dir = Path("tmp/unzipped_l2")
unzipped_dir.mkdir(parents=True, exist_ok=True)

zip_files = sorted(list(raw_zip_dir.glob("*.zip")))

print(f"Extraction of {len(zip_files)} archives starting...")

for zp in tqdm(zip_files):
    with zipfile.ZipFile(zp, 'r') as z:
        z.extractall(unzipped_dir)

print("Extraction complete.")
