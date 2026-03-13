"""
path_utils.py — QuantGod Path Resolution Utilities
====================================================
Single source of truth for all local and remote paths.

Convention:
  LOCAL  → generic names, no run params (e.g. data/L2/pre_processed)
  DRIVE  → all exports go under a single session root per pipeline run:

  drive:PROJETOS/RESULTADOS_SELL_{s}_BUY_{b}_{mins}min/               ← [MASTER BUCKET]
  ├── MODELOS_06_03_2026_v001/    ← transfer.py        (PTs + scalers + optuna.db)
  ├── AUDITORIA_06_03_2026_v001/  ← upload_audit_to_drive
  │   ├── ETL/
  │   ├── LABELLING/
  │   └── KFOLD_SPECIALIST/
  ├── PRE_PROCESSED_...           ← run_pipeline.py    (Parquets sem target)
  └── LABELLED_...                ← run_labelling.py   (Parquets com target)

  As subpastas (MODELOS, AUDITORIA) são versionadas pelo timestamp da sessão,
  enquanto a raiz é compartilhada por todos os experimentos do mesmo setup (SELL/BUY).

  Os splits locais (labelled/train, labelled/val, specialized/) NAO sobem para o Drive.
  Eles são derivados que podem ser recriados a partir de LABELLED + split_dataset.py.

  O session timestamp é fixado uma vez por processo via _get_session_timestamp() (lru_cache),
  garantindo que TODAS as etapas de uma mesma execução compartilhem a mesma pasta raiz.
"""
from pathlib import Path
from datetime import datetime
import functools


# ─────────────────────────────────────────────────────────────────────────────
# Session timestamp (fixed once per process)
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_session_timestamp() -> str:
    """Returns a fixed timestamp for the current process session (YYMMdd_HHmmss)."""
    return datetime.now().strftime("%y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# Drive suffix & session root helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_drive_suffix(config: dict) -> str:
    """
    Builds the run-specific suffix appended to the Drive session root folder.
    Format: _SELL_{s}_BUY_{b}_{t}min_lookahead_{freq}
    """
    sell = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    t = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    
    freq = config['pre_processing']['etl'].get('resample_freq', "1min")
    
    # Remove dots from thresholds as requested
    sell_str = str(sell).replace(".", "")
    buy_str  = str(buy).replace(".", "")
    
    return f"_SELL_{sell_str}_BUY_{buy_str}_{t}min_lookahead_{freq}_bar"


def get_drive_session_root(config: dict) -> str:
    """
    Returns the threshold-based parent folder on Drive.
    Format: drive:PROJETOS/RESULTADOS_SELL_{s}_BUY_{b}_{mins}min
    This folder serves as a 'bucket' for all runs sharing the same labelling logic.
    """
    base = config['pipeline_paths'].get('drive_results_root', 'drive:PROJETOS/RESULTADOS')
    suffix = get_drive_suffix(config)
    return f"{base}{suffix}"


def get_drive_session_path(subfolder: str, config: dict) -> str:
    """
    Returns the full Drive path for a given subfolder, grouping all stages
    under a single session timestamp directory.
    Example: get_drive_session_path("MODELOS", cfg)
             → "drive:PROJETOS/RESULTADOS_.../06_03_2026_v001/MODELOS"
    """
    root = get_drive_session_root(config)
    
    cfg_ts = config.get('pipeline_paths', {}).get('session_timestamp')
    ts = str(cfg_ts).strip() if cfg_ts else _get_session_timestamp()
    
    # Group all subfolders under the session timestamp folder
    return f"{root}/{ts}/{subfolder}"


# ─────────────────────────────────────────────────────────────────────────────
# Local path helpers (generic names, no run params)
# ─────────────────────────────────────────────────────────────────────────────

def _l2_root(config: dict) -> Path:
    return Path(config['pipeline_paths'].get('local_data_root', 'data/L2'))


def get_temp_raw_dir(config: dict) -> str:
    """Returns: data/L2/temp_raw"""
    sub = config['pipeline_paths'].get('temp_raw_dir', 'temp_raw')
    return str(_l2_root(config) / sub)


def get_pre_processed_dir(config: dict) -> str:
    """Returns: data/L2/pre_processed"""
    sub = config['pipeline_paths'].get('pre_processed_dir', 'pre_processed')
    return str(_l2_root(config) / sub)


def get_labelled_dir(config: dict) -> str:
    """Returns: data/L2/labelled"""
    sub = config['pipeline_paths'].get('labelled_dir', 'labelled')
    return str(_l2_root(config) / sub)


def get_specialized_dir(config: dict) -> str:
    """Returns: data/L2/specialized"""
    sub = config['pipeline_paths'].get('specialized_dir', 'specialized')
    return str(_l2_root(config) / sub)


def get_logs_root(config: dict) -> str:
    """Returns: logs"""
    return config['pipeline_paths'].get('local_logs_root', 'logs')


def get_reports_root(config: dict) -> str:
    """Returns: docs/reports"""
    return config['pipeline_paths'].get('local_reports_root', 'docs/reports')


def get_fused_dataset_dir(config: dict) -> str:
    """Returns: data/auditor/dataset_fused"""
    return config['pipeline_paths'].get('fused_dataset_dir', 'data/auditor/dataset_fused')


def get_auditor_context_dir(config: dict) -> str:
    """Returns: data/auditor/context"""
    return config['pipeline_paths'].get('auditor_context_dir', 'data/auditor/context')


def get_auditor_oof_dir(config: dict) -> str:
    """Returns: data/auditor/oof_predictions"""
    return config['pipeline_paths'].get('auditor_oof_dir', 'data/auditor/oof_predictions')


# ─────────────────────────────────────────────────────────────────────────────
# Legacy compat shim (kept for any code that still calls get_drive_dir directly)
# ─────────────────────────────────────────────────────────────────────────────

def get_drive_dir(base_remote: str, config: dict) -> str:
    """
    DEPRECATED — use get_drive_session_path() or get_drive_session_root() instead.
    Redirects legacy calls to the new Threshold Bucket root.
    """
    from .logging_utils import logger
    logger.warning("get_drive_dir is deprecated. Using get_drive_session_path instead.")
    return get_drive_session_path("MODELOS", config)

def resolve_local_drive(path_str) -> Path:
    """
    Converts a rclone-style 'drive:...' path into a local filesystem path.
    - Windows Local: Resolves to 'G:/Meu Drive' if 'G:' existe.
    - Cloud/Linux/No G:: Resolve para o diretório de trabalho atual (pasta do projeto).
    """
    import os
    import sys
    path_str = str(path_str)
    
    if "drive:" in path_str:
        _, subpath = path_str.split("drive:", 1)
        subpath = subpath.lstrip("\\/")
        
        # Se contiver o prefixo PROJETOS/, removemos para o path local ficar limpo
        if subpath.upper().startswith("PROJETOS/"):
            subpath = subpath[len("PROJETOS/"):]

        # Se for Windows e existir o drive G:, usamos ele.
        # Caso contrário (Linux/Cloud), usamos a raiz do projeto.
        g_drive = Path("G:/Meu Drive")
        if sys.platform == "win32" and g_drive.exists():
            return g_drive / subpath
        
        # Fallback para Cloud (RunPod/Lambda) ou sistemas sem G:
        return Path.cwd() / subpath
            
    return Path(path_str)


def resolve_local_project(path_str, project_root: Path) -> Path:
    """
    Resolves a rclone-style 'drive:PROJETOS/RESULTADOS_...' path to the local
    RESULTADOS_... folder that already exists at the project root.

    This is used exclusively by the Paper Trading system so that it reads
    models & configs from the project directory instead of Google Drive.
    
    Convention: drive:PROJETOS/RESULTADOS_X/Y → <project_root>/RESULTADOS_X/Y

    If the path does not include a Drive prefix, falls back to a plain Path.
    """
    path_str = str(path_str).replace("\\", "/")

    if "drive:" in path_str:
        _, subpath = path_str.split("drive:", 1)
        # Remove leading 'PROJETOS/' prefix that lives on the Drive root
        subpath = subpath.lstrip("/")
        if subpath.upper().startswith("PROJETOS/"):
            subpath = subpath[len("PROJETOS/"):]
        return project_root / subpath

    return project_root / path_str if not Path(path_str).is_absolute() else Path(path_str)
