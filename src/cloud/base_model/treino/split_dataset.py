import os
import shutil
from pathlib import Path

def split_labelled_data():
    source_dir = Path("data/L2/labelled_test_SELL_0002_BUY_0002_1h")
    if not source_dir.exists():
        print(f"Directory not found: {source_dir}")
        return

    # Extract suffix from source_dir name (e.g. labelled_test_SELL_0002_BUY_0002_1h -> SELL_0002_BUY_0002_1h)
    suffix_match = source_dir.name.split("test_")[-1] if "test_" in source_dir.name else source_dir.name
    
    # Base directory for splits with suffix
    split_base = Path(f"data/L2/splits_{suffix_match}")
    train_dir = split_base / "train"
    val_dir = split_base / "val"
    test_dir = split_base / "test"

    # Clean previous splits if they exist
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(list(source_dir.glob("*.parquet")))
    total_files = len(files)
    
    if total_files == 0:
        print("No parquet files found.")
        return

    # Ratios: 70/20/10
    n_train = int(total_files * 0.7)
    n_val = int(total_files * 0.2)
    # The rest goes to test
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"Total files: {total_files}")
    print(f"Moving {len(train_files)} to train...")
    for f in train_files:
        shutil.copy(f, train_dir / f.name)

    print(f"Moving {len(val_files)} to val...")
    for f in val_files:
        shutil.copy(f, val_dir / f.name)

    print(f"Moving {len(test_files)} to test...")
    for f in test_files:
        shutil.copy(f, test_dir / f.name)

    print("\n✅ Split completed:")
    print(f"  Train: {train_dir} ({len(train_files)} files)")
    print(f"  Val:   {val_dir} ({len(val_files)} files)")
    print(f"  Test:  {test_dir} ({len(test_files)} files)")

if __name__ == "__main__":
    split_labelled_data()
