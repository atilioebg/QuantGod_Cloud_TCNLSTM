import logging
import sys
import io
import os
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
            msg = self.format(record)
            stream = self.stream
            # Robustly append newline if missing to avoid merged lines with tqdm
            if not msg.endswith('\n'):
                msg += '\n'
            stream.write(msg)
            self.flush()
        except OSError:
            pass

    def flush(self):
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
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
        # Avoid double wrapping or issues in non-standard environments
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
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
    
    # Hide "INITIALIZED" message if QUIET_LOGGING is set (for cleaner subprocesses)
    if not os.environ.get('QUIET_LOGGING'):
        # Leading newline (\n) ensures we don't merge with the terminal prompt
        logger.info(f"\n📝 LOGGING INITIALIZED (UTF-8-SIG): {log_file}")
        
    return logger

def setup_optuna_logging(logger_name: str = "optuna"):
    """
    Standardizes Optuna logging to use our formatting and prevent duplicates.
    """
    import optuna
    # 1. Disable Optuna's default console handler to avoid double logging
    optuna.logging.disable_default_handler()
    # 2. Re-enable propagation so Optuna records reach our root logger (and our handlers)
    optuna.logging.enable_propagation()
    # 3. Set standard info level
    optuna.logging.set_verbosity(optuna.logging.INFO)
    return logging.getLogger(logger_name)

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


def upload_audit_to_drive(
    local_dirs: list,
    stage_name: str,
    extra_files: list = None,
    rclone_config: str = "rclone.conf",
    config: dict = None,
):
    """
    v4.9+: Uploads audit files (logs + reports) to Google Drive.

    Remote destination:
      - With config: drive:PROJETOS/AUDITORIA_SELL_..._min/{stage_name}/
      - Without config (fallback): drive:PROJETOS/AUDITORIA/{stage_name}/

    Args:
        local_dirs:    List of local directory paths (str or Path) to upload recursively.
                       Non-existent directories are silently skipped.
        stage_name:    Pipeline stage identifier used as Drive subfolder  (e.g. "ETL",
                       "LABELLING", "KFOLD_SPECIALIST", "AUDITOR").
        extra_files:   Optional list of individual files to copy. Each file is uploaded
                       to the stage's root folder on Drive.
        rclone_config: Path to rclone.conf (default: "rclone.conf" in the project root).
        config:        master_config dict (optional). When supplied, the Drive destination
                       includes the run-specific suffix (SELL/BUY/lookahead).

    Notes:
        - Never raises exceptions — errors are logged and the function returns silently.
        - Windows fallback: prefers rclone.exe in project root when on Windows.
        - Uses `rclone copy` (not sync) to preserve existing Drive content.
    """
    import subprocess as _sp
    import os as _os

    logger = logging.getLogger(__name__)

    # Resolve rclone binary (Windows-aware)
    rclone_bin = "rclone"
    if _os.name == "nt" and Path("rclone.exe").exists():
        rclone_bin = str(Path("rclone.exe").absolute())

    cfg_args = ["--config", rclone_config] if Path(rclone_config).exists() else []

    # Build remote base: centralized session root + stage subfolder
    if config is not None:
        try:
            from src.cloud.base_model.utils.path_utils import get_drive_session_path
            remote_base = get_drive_session_path(f"AUDITORIA/{stage_name.upper()}", config)
        except Exception:
            remote_base = f"drive:PROJETOS/RESULTADOS/AUDITORIA/{stage_name.upper()}"
    else:
        remote_base = f"drive:PROJETOS/RESULTADOS/AUDITORIA/{stage_name.upper()}"

    logger.info(f"📤 [AUDIT UPLOAD] Stage={stage_name} → {remote_base}")

    # 1. Upload each directory
    for local_dir in (local_dirs or []):
        src = Path(local_dir)
        if not src.exists() or not src.is_dir():
            logger.debug(f"  ↳ Skipping (not found): {src}")
            continue
        # Per-directory sub-path mirrors the local name for organisation
        remote_dest = f"{remote_base}/{src.name}"
        cmd = [rclone_bin, "copy", str(src), remote_dest, "-P"] + cfg_args
        try:
            _sp.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"  ✅ Uploaded {src} → {remote_dest}")
        except _sp.CalledProcessError as e:
            logger.error(f"  ❌ Failed to upload {src}: {e.stderr.strip()}")
        except Exception as e:
            logger.error(f"  ❌ Upload error for {src}: {e}")

    # 2. Upload individual extra files (e.g. docs/reports/*.json)
    for file_path in (extra_files or []):
        fp = Path(file_path)
        if not fp.exists() or not fp.is_file():
            logger.debug(f"  ↳ Skipping file (not found): {fp}")
            continue
        cmd = [rclone_bin, "copyto", str(fp), f"{remote_base}/{fp.name}"] + cfg_args
        try:
            _sp.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"  ✅ Uploaded file {fp.name} → {remote_base}/")
        except _sp.CalledProcessError as e:
            logger.error(f"  ❌ Failed to upload file {fp}: {e.stderr.strip()}")
        except Exception as e:
            logger.error(f"  ❌ Upload error for file {fp}: {e}")

    logger.info(f"📤 [AUDIT UPLOAD] {stage_name} complete.")
