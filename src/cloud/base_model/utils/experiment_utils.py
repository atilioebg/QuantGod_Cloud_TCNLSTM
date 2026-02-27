import yaml
import logging
from pathlib import Path
import os
import re
from src.cloud.base_model.utils.logging_utils import get_labelling_suffix

logger = logging.getLogger(__name__)



def resolve_active_labelled_dir() -> Path:
    """
    Determines the correct labelled directory based on labelling_config.yaml.
    Falls back to the most recent labelled_* folder if the config file is
    missing, empty, or deprecated (contains only comments → YAML parses to None).
    """
    config_path = Path("src/cloud/base_model/labelling/labelling_config.yaml")
    if not config_path.exists():
        # Fallback to most recent folder if config missing
        base = Path("data/L2")
        dirs = sorted(list(base.glob("labelled_*")))
        return dirs[-1] if dirs else base / "labelled"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Guard: deprecated file (only comments) parses to None — use fallback
    if config is None or 'params' not in config or 'paths' not in config:
        base = Path("data/L2")
        dirs = sorted(list(base.glob("labelled_*")))
        return dirs[-1] if dirs else base / "labelled"

    suffix = get_labelling_suffix(config['params'])
    base_output = Path(config['paths']['output_dir'])

    # If the base output dir doesn't already end with the suffix, append it
    if not base_output.name.endswith(suffix):
        return base_output.parent / f"{base_output.name}{suffix}"
    return base_output

def resolve_data_paths(config_paths: dict) -> tuple:
    """
    Resolves train_dir and val_dir. If set to "AUTO", it uses the active labelled directory.
    Returns (train_path, val_path).
    """
    train_dir = config_paths.get('train_dir')
    val_dir = config_paths.get('val_dir')
    
    if train_dir == "AUTO" or val_dir == "AUTO":
        active_dir = resolve_active_labelled_dir()
        
        # Check if it should be in a splits subfolder (like splits_labelled_.../train)
        # We check for the splits variant first
        splits_dir = active_dir.parent / f"splits_{active_dir.name}"
        
        if splits_dir.exists():
            resolved_train = splits_dir / "train"
            resolved_val = splits_dir / "val"
        else:
            # 🚨 CRITICAL AUDIT FIX: Never fallback to the same folder for Train and Val.
            # This causes massive data leakage where the model tests on its training data.
            raise FileNotFoundError(
                f"\n❌ CRITICAL ERROR: Data split directory not found at {splits_dir}.\n"
                f"You MUST run 'python src/cloud/base_model/treino/split_dataset.py' "
                f"before running Optuna to create chronological train/val splits and "
                f"prevent data leakage!"
            )
            
        return str(resolved_train), str(resolved_val)
    
    return train_dir, val_dir
