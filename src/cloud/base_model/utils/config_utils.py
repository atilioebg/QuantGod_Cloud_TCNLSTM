import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str = "src/cloud/base_model/configs/master_config.yaml"):
    """
    Loads the master YAML configuration file.
    Default path: src/cloud/base_model/configs/master_config.yaml
    """
    path = Path(config_path)
    if not path.exists():
        # Try to find it from project root if running from subdirectories
        # (This is a safety check for different execution contexts)
        possible_paths = [
            Path("src/cloud/base_model/configs/master_config.yaml"),
            Path("../../src/cloud/base_model/configs/master_config.yaml"),
            Path("../../../src/cloud/base_model/configs/master_config.yaml"),
            Path("../../../../src/cloud/base_model/configs/master_config.yaml"),
        ]
        for p in possible_paths:
            if p.exists():
                path = p
                break
    
    if not path.exists():
        raise FileNotFoundError(f"❌ Configuration file not found: {config_path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
