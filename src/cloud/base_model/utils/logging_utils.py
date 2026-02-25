import logging
import sys
import io
from pathlib import Path
from datetime import datetime

class RobustStreamHandler(logging.StreamHandler):
    """
    A StreamHandler that silently swallows OSError / IOError on emit and flush.
    This prevents a detached SSH terminal (Errno 5: I/O error) from crashing a
    long-running cloud process (e.g., Optuna optimization on RunPod).
    The FileHandler is unaffected and continues writing to disk normally.
    """
    def emit(self, record):
        try:
            super().emit(record)
        except OSError:
            pass

    def flush(self):
        try:
            super().flush()
        except OSError:
            pass


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
    
    # ── UTF-8 Terminal Fix (Windows/CI Compatibility) ────────────────────────
    # We wrap stdout to ensure it handles UTF-8 even if the system default is different
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            RobustStreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w', encoding='utf-8-sig')
        ],
        force=True
    )
    logger = logging.getLogger(log_module_name)
    logger.info(f"📝 LOGGING INITIALIZED (UTF-8-SIG): {log_file}")
    return logger

def get_labelling_suffix(params: dict) -> str:
    """
    Generates standard suffix: _SELL_0003_BUY_0003_15min
    Supports both 1min pipelines (lookahead in minutes) and 5min pipelines
    (lookahead in bars). Uses 'bar_size_min' param (default=5) to convert.
    """
    s_val = int(round(abs(params.get('threshold_short', 0)) * 1000))
    b_val = int(round(abs(params.get('threshold_long', 0)) * 1000))

    bar_size_min = params.get('bar_size_min', 5)  # default: 5min bars (Sniper Pivot)
    total_minutes = int(params.get('lookahead', 3)) * bar_size_min

    if total_minutes >= 60 and total_minutes % 60 == 0:
        time_label = f"{total_minutes // 60}h"
    else:
        time_label = f"{total_minutes}min"

    return f"_SELL_{s_val:04d}_BUY_{b_val:04d}_{time_label}"
