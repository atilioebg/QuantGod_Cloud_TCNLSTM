import os
import shutil
from pathlib import Path
import sys

def split_labelled_data(source_dir_path=None):
    if source_dir_path:
        source_dir = Path(source_dir_path)
    elif len(sys.argv) > 1:
        source_dir = Path(sys.argv[1])
    else:
        # Fallback logic: check for folders starting with 'labelled' in data/L2
        base_path = Path("data/L2")
        if not base_path.exists():
            print(f"Base path not found: {base_path}")
            return
            
        labelled_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("labelled")], key=lambda x: x.stat().st_mtime, reverse=True)
        if labelled_dirs:
            source_dir = labelled_dirs[0]
            print(f"No source directory provided. Using most recent: {source_dir}")
        else:
            print("No labelled directories found in data/L2")
            return

    if not source_dir.exists():
        print(f"Directory not found: {source_dir}")
        return

    # Suffix logic - use the directory name itself as suffix for the split folder
    suffix = source_dir.name
    
    # Base directory for splits with suffix
    split_base = Path(f"data/L2/splits_{suffix}")
    train_dir = split_base / "train"
    val_dir = split_base / "val"
    test_dir = split_base / "test"

    print(f"Source: {source_dir}")
    print(f"Target split base: {split_base}")

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
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

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
