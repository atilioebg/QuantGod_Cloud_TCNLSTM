import pandas as pd
from pathlib import Path

def get_pre_processed_dir(config: dict) -> str:
    """
    Retorna o caminho local canônico de saída do ETL.
    Formato: {root}/{prefix}_{horizon}_{lookback}_{freq}
    """
    res_freq     = config['pre_processing']['etl'].get('resample_freq', '5min')
    res_min      = "".join(filter(str.isdigit, res_freq)) or "5"
    horizon_min  = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    lookback_min = config['pre_processing']['etl'].get('lookback_minutes', 120)
    
    root   = config['pipeline_paths'].get('local_data_root', 'data/L2')
    prefix = config['pipeline_paths'].get('pre_processed_prefix', 'PRE_PROCESSED_L2')
    return str(Path(root) / f"{prefix}_{horizon_min}_{lookback_min}_{res_min}")

def get_labelled_dir(config: dict) -> str:
    """
    Retorna o caminho local de saída do Labelling.
    Formato: {root}/{prefix}_SELL_{sell}_BUY_{buy}_{mins}min
    """
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    
    root   = config['pipeline_paths'].get('local_data_root', 'data/L2')
    prefix = config['pipeline_paths'].get('labelled_prefix', 'splits_labelled')
    suffix = f"_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    return str(Path(root) / f"{prefix}{suffix}")

def get_specialized_dir(config: dict) -> str:
    """
    Retorna o caminho local de saída do Especialista.
    Formato: {root}/{prefix}_SELL_{sell}_BUY_{buy}_{mins}min
    """
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    
    root   = config['pipeline_paths'].get('local_data_root', 'data/L2')
    prefix = config['pipeline_paths'].get('specialized_prefix', 'splits_specialized_labelled')
    suffix = f"_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    return str(Path(root) / f"{prefix}{suffix}")

def get_temp_raw_dir(config: dict) -> str:
    """Retorna o diretório temporário para extração."""
    root = config['pipeline_paths'].get('local_data_root', 'data/L2')
    temp = config['pipeline_paths'].get('temp_raw_dir', 'temp_raw')
    return str(Path(root) / temp)
