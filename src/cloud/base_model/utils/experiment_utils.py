import yaml
import logging
from pathlib import Path
import os
import re

logger = logging.getLogger(__name__)



from src.cloud.base_model.utils.path_utils import get_labelled_dir

def resolve_active_labelled_dir() -> Path:
    """
    Determines the correct labelled directory based on master_config.yaml.
    """
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    if not master_cfg_path.exists():
        # Fallback to most recent folder if config missing
        base = Path("data/L2")
        dirs = sorted(list(base.glob("labelled_*")))
        return dirs[-1] if dirs else base / "labelled"

    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # In modern v5 architecture, the local folder is not suffixed.
    # The suffix only applies to the Google Drive session export.
    return Path(get_labelled_dir(config))

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
        
        # Modern v5.0 structure saves train/val inside the active directory itself
        if (active_dir / "train").exists() and (active_dir / "val").exists():
            resolved_train = active_dir / "train"
            resolved_val = active_dir / "val"
        # Legacy v4 structure fallback
        elif splits_dir.exists():
            resolved_train = splits_dir / "train"
            resolved_val = splits_dir / "val"
        else:
            # 🚨 CRITICAL AUDIT FIX: Never fallback to the same folder for Train and Val.
            # This causes massive data leakage where the model tests on its training data.
            raise FileNotFoundError(
                f"\n❌ CRITICAL ERROR: Data split directory not found at {active_dir}/train or {splits_dir}.\n"
                f"You MUST run 'python src/cloud/base_model/treino/split_dataset.py' "
                f"before running Optuna to create chronological train/val splits and "
                f"prevent data leakage!"
            )
            
        return str(resolved_train), str(resolved_val)
    
    return train_dir, val_dir
