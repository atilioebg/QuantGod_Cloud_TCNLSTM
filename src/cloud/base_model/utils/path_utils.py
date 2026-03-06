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
    Format: _SELL_{sell_val}_BUY_{buy_val}_{horizon}min
    Example: _SELL_00030_BUY_00030_15min
    """
    sell = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    sell_str = f"{int(round(sell * 10000)):05d}"
    buy_str  = f"{int(round(buy  * 10000)):05d}"
    return f"_SELL_{sell_str}_BUY_{buy_str}_{mins}min"


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
    Returns the full Drive path for a given subfolder, injecting the session timestamp.
    Example: get_drive_session_path("MODELOS", cfg)
             → "drive:PROJETOS/RESULTADOS_SELL_..._min/MODELOS_06_03_2026_v001"
    """
    root = get_drive_session_root(config)
    
    cfg_ts = config.get('pipeline_paths', {}).get('session_timestamp')
    ts = str(cfg_ts).strip() if cfg_ts else _get_session_timestamp()
    
    # Inject timestamp to the first folder component to group properly
    # e.g., "AUDITORIA/ETL" -> "AUDITORIA_06_03_2026_v001/ETL"
    parts = subfolder.split('/')
    parts[0] = f"{parts[0]}_{ts}"
    new_sub = "/".join(parts)
    
    return f"{root}/{new_sub}"


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
    return get_drive_session_root(config)
