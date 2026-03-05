"""
path_utils.py — QuantGod Path Resolution Utilities
====================================================
Single source of truth for all local and remote paths.

Convention:
  LOCAL  → generic names, no run params (e.g. data/L2/pre_processed)
  DRIVE  → all exports go under a single session root per pipeline run:

  drive:PROJETOS/RESULTADOS_SELL_{s}_BUY_{b}_{mins}min_{timestamp}/
  ├── PRE_PROCESSED/          ← run_pipeline.py        (ETL output: parquets sem target)
  ├── LABELLED/               ← run_labelling.py       (Labelling output: parquets com coluna 'target')
  ├── AUDITORIA/
  │   ├── ETL/                ← run_pipeline.py        (logs + data_quality_report.json + audit_summary.csv)
  │   ├── LABELLING/          ← run_labelling.py       (logs + labelling_health_QA.log)
  │   ├── SPLIT/              ← split_dataset.py       (logs + split_summary.json)
  │   ├── KFOLD_SPECIALIST/   ← run_kfold_specialist.py (logs + kfold_security_QA.log)
  │   └── AUDITOR/            ← train_xgboost.py       (logs + feature_importance.json)
  └── MODELOS/
      ├── foundation/         ← transfer.py            (best_tcn_lstm.pt + scaler.pkl + optuna.db)
      └── specialized/        ← transfer.py            (best_specialist.pt + kfold scalers + oof.parquet)

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
    Returns the unique session root folder on Drive for this pipeline run.
    Format: drive:PROJETOS/RESULTADOS_SELL_{s}_BUY_{b}_{mins}min_{timestamp}
    All stages (ETL, Labelling, Audit, Models) share this same root.
    """
    base = config['pipeline_paths'].get('drive_results_root', 'drive:PROJETOS/RESULTADOS')
    suffix = get_drive_suffix(config)
    
    # Check if a fixed timestamp was provided in master_config.yaml
    cfg_ts = config.get('pipeline_paths', {}).get('session_timestamp')
    if cfg_ts:
        ts = str(cfg_ts).strip()
    else:
        # Fallback to generation if not defined
        ts = _get_session_timestamp()
        
    return f"{base}{suffix}_{ts}"


def get_drive_session_path(subfolder: str, config: dict) -> str:
    """
    Returns the full Drive path for a given subfolder within the session root.
    Example: get_drive_session_path("AUDITORIA/ETL", cfg)
             → "drive:PROJETOS/RESULTADOS_SELL_00030_BUY_00030_15min_260304_192238/AUDITORIA/ETL"
    """
    return f"{get_drive_session_root(config)}/{subfolder}"


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
    DEPRECATED — use get_drive_session_path() instead.
    Kept for backward compatibility with scripts not yet migrated.
    Appends only the param suffix (no timestamp) to a given base remote.
    """
    return f"{base_remote}{get_drive_suffix(config)}"
