import polars as pl
import yaml
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
import gc
import zipfile
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parents[2].absolute()
sys.path.append(str(PROJECT_ROOT))

# Imports
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.labelling.run_labelling import process_single_file_labelling

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BACKTEST_DATA - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_backtest_config():
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    # Detect environment: Cloud (/workspace) vs Local
    CLOUD_CONFIG = PROJECT_ROOT / "src" / "backtest" / "backtest_config_cloud.yaml"
    LOCAL_CONFIG = PROJECT_ROOT / "src" / "backtest" / "backtest_config.yaml"

    CONFIG_PATH = CLOUD_CONFIG if CLOUD_CONFIG.exists() and "/workspace" in str(PROJECT_ROOT) else LOCAL_CONFIG
    master_config_path = PROJECT_ROOT / "src/cloud/base_model/configs/master_config.yaml"
    
    with open(master_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        bt_config = yaml.safe_load(f)
        
    # Merge/Override paths from backtest config
    config['pipeline_paths'].update(bt_config['pipeline_paths'])
    return config

def process_single_day_etl(zip_path, raw_trade_dir, pre_processed_dir, config):
    date_str = zip_path.name.split("_")[0]
    trade_path = raw_trade_dir / f"BTCUSDT{date_str}.csv.gz"
    out_name = zip_path.stem.replace(".data", "") + ".parquet"
    out_path = pre_processed_dir / out_name
    
    # Skip if already exists
    if out_path.exists():
        logger.info(f"⏩ [SKIP] {out_name} already exists.")
        return {"file": zip_path.name, "status": "skipped"}

    if not trade_path.exists():
        return {"file": zip_path.name, "status": "missing_trades"}
        
    unzipped_l2_dir = PROJECT_ROOT / "tmp" / "unzipped_l2"
    
    # 1. Process Orderbook (L2) messages from unzipped file
    l2_file_path = unzipped_l2_dir / f"{date_str}_BTCUSDT_ob200.data"
    if not l2_file_path.exists():
        return {"file": zip_path.name, "status": "missing_unzipped_l2"}

    try:
        transformer = L2Transformer(
            levels=config['pre_processing']['etl']['levels'],
            sampling_ms=config['pre_processing']['etl']['sampling_ms'],
            etl_cfg=config['pre_processing']['etl']
        )
        
        rows = []
        with open(l2_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    res = transformer.process_message(msg)
                    if res: rows.append(res)
                except: continue
        
        if not rows:
            return {"file": zip_path.name, "status": "no_data"}
            
        df_l2 = pl.DataFrame(rows)
        # Clear rows memory
        del rows
        gc.collect()

        df_trades = pl.read_csv(trade_path)
        
        df_trades = df_trades.with_columns([
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64).alias("amount"),
            pl.when(pl.col("side") == "Buy").then(1).otherwise(-1).alias("side")
        ])
        
        # Interleave L2 and Trades correctly for Sampler
        df_combined = df_l2.join(df_trades.select(['ts', 'amount', 'side']), on='ts', how='full').sort('ts').fill_null(strategy='forward')
        
        # Free memory L2/Trades
        del df_l2, df_trades
        gc.collect()

        df_final = transformer.apply_feature_engineering(df_combined)
        del df_combined
        gc.collect()

        df_final.write_parquet(out_path)
        del df_final
        gc.collect()
        
        return {"file": zip_path.name, "status": "ok"}
    except Exception as e:
        return {"file": zip_path.name, "status": "error", "message": str(e)}

def run_backtest_pipeline_robust(config):
    raw_zip_dir = Path(config['pipeline_paths']['local_data_root']) / "raw_zip"
    raw_trade_dir = Path(config['pipeline_paths']['local_data_root']) / "raw_trades"
    pre_processed_dir = Path(config['pipeline_paths']['local_data_root']) / "pre_processed"
    labelled_dir = Path(config['pipeline_paths']['local_data_root']) / "labelled"
    
    pre_processed_dir.mkdir(parents=True, exist_ok=True)
    labelled_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = sorted(list(raw_zip_dir.glob("*.zip")))
    
    # ── 1. Sequential ETL (Robust on Windows) ──
    logger.info(f"🚀 Starting ROBUST Sequential Backtest ETL for {len(zip_files)} days...")
    for zp in tqdm(zip_files, desc="ETL Work"):
        res = process_single_day_etl(zp, raw_trade_dir, pre_processed_dir, config)
        if res['status'] == 'error':
            logger.error(f"❌ {res['file']}: {res.get('message', '')}")
        gc.collect()

    # ── 2. Sequential Labelling (Robust on Windows) ──
    pre_files = sorted(list(pre_processed_dir.glob("*.parquet")))
    logger.info(f"🚀 Starting ROBUST Sequential Backtest Labelling for {len(pre_files)} files...")
    for pf in tqdm(pre_files, desc="Labelling Work"):
        # Skip if labelled already exists
        out_lab = labelled_dir / pf.name
        if out_lab.exists():
            continue
            
        res = process_single_file_labelling(pf, config, labelled_dir)
        if "error" in res:
            logger.error(f"❌ Error labelling {res.get('file')}: {res['error']}")
        gc.collect()

if __name__ == "__main__":
    config = load_backtest_config()
    run_backtest_pipeline_robust(config)
    logger.info("✅ Integrated Backtest Data Pipeline Complete.")
