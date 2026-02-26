import os
import shutil
from pathlib import Path
import sys
import yaml

def split_labelled_data(source_dir_path=None):
    # Load Master Config
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Reconstruct exact expected output dir from config values
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    suffix = f"_labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    base_split_dir_name = f"splits{suffix}"

    if source_dir_path:
        source_dir = Path(source_dir_path)
    elif len(sys.argv) > 1:
        source_dir = Path(sys.argv[1])
    else:
        # Expected from the new config flow
        source_dir = Path(f"data/L2/{base_split_dir_name}")
        
    if not source_dir.exists():
        print(f"Directory not found: {source_dir}")
        return

    # Suffix logic - use the directory name itself as suffix for the split folder
    suffix = source_dir.name
    
    # Base directory for splits with suffix
    split_base = Path(f"data/L2/{base_split_dir_name}")
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

    train_pct = config['pre_processing']['split']['base'].get('train_ratio', 0.85)
    val_pct = config['pre_processing']['split']['base'].get('val_ratio', 0.15)

    # Ratios
    n_train = int(total_files * train_pct)
    n_val = int(total_files * val_pct)
    
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

    print("\nSplit completed:")
    print(f"  Train: {train_dir} ({len(train_files)} files)")
    print(f"  Val:   {val_dir} ({len(val_files)} files)")
    print(f"  Test:  {test_dir} ({len(test_files)} files)")

if __name__ == "__main__":
    split_labelled_data()
