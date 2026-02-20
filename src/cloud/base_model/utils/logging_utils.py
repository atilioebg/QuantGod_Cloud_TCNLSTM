import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(log_module_name: str, suffix: str = ""):
    """
    Standardized logger setup for all QuantGod modules.
    Creates a log file in logs/{log_module_name}/{log_module_name}_{suffix}_{timestamp}.log
    """
    log_dir = Path(f"logs/{log_module_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Remove existing handlers to avoid duplicates during interactive sessions
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    log_file = log_dir / f"{log_module_name}{suffix}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ],
        force=True
    )
    logger = logging.getLogger(log_module_name)
    logger.info(f"📝 LOGGING INITIALIZED: {log_file}")
    return logger

def get_labelling_suffix(params: dict) -> str:
    """
    Generates standard suffix: _SELL_0002_BUY_0002_1h
    used for both folder names and log names.
    """
    s_val = int(round(abs(params.get('threshold_short', 0)) * 1000))
    b_val = int(round(abs(params.get('threshold_long', 0)) * 1000))
    h_val = int(params.get('lookahead', 60) / 60)
    
    return f"_SELL_{s_val:04d}_BUY_{b_val:04d}_{h_val}h"
