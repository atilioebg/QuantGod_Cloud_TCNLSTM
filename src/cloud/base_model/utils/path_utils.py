"""
path_utils.py — QuantGod Path Resolution Utilities
====================================================
Single source of truth for all local and remote paths.

Convention:
  LOCAL  → generic names, no run params (e.g. data/L2/pre_processed)
  DRIVE  → suffix with SELL/BUY/lookahead params appended at runtime
             (e.g. drive:PROJETOS/PRE_PROCESSED_L2_SELL_00030_BUY_00030_15min)
"""
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Drive suffix helper
# ─────────────────────────────────────────────────────────────────────────────

def get_drive_suffix(config: dict) -> str:
    """
    Builds the run-specific suffix appended to ALL Drive folder names.
    Format: _SELL_{sell_val}_BUY_{buy_val}_{horizon}min
    Example: _SELL_00030_BUY_00030_15min
    """
    sell = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    sell_str = f"{int(round(sell * 10000)):05d}"
    buy_str  = f"{int(round(buy  * 10000)):05d}"
    return f"_SELL_{sell_str}_BUY_{buy_str}_{mins}min"


def get_drive_dir(base_remote: str, config: dict) -> str:
    """
    Appends the run suffix to a Drive base remote path.
    Example: get_drive_dir("drive:PROJETOS/PRE_PROCESSED_L2", cfg)
             → "drive:PROJETOS/PRE_PROCESSED_L2_SELL_00030_BUY_00030_15min"
    """
    return f"{base_remote}{get_drive_suffix(config)}"


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
    """Returns: data/L2/splits_labelled"""
    sub = config['pipeline_paths'].get('labelled_dir', 'splits_labelled')
    return str(_l2_root(config) / sub)


def get_specialized_dir(config: dict) -> str:
    """Returns: data/L2/splits_specialized_labelled"""
    sub = config['pipeline_paths'].get('specialized_dir', 'splits_specialized_labelled')
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
