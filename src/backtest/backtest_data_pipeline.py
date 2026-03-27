import polars as pl
import yaml
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess
import gc
import zipfile
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    logger.info(f"🔄 Loading Backtest Config from: {CONFIG_PATH}")
    master_config_path = PROJECT_ROOT / "src/cloud/base_model/configs/master_config.yaml"
    
    with open(master_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        bt_config = yaml.safe_load(f)
        
    # Merge/Override all top-level keys from backtest config (pipeline_paths, processing, simulation)
    for k, v in bt_config.items():
        if k in config and isinstance(config[k], dict) and isinstance(v, dict):
            config[k].update(v)
        else:
            config[k] = v
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
    l2_file_path = unzipped_l2_dir / f"{date_str}_BTCUSDT_ob200.data"
    
    # 1. Process Orderbook (L2) messages - Try unzipped first, fallback to ZIP
    rows = {}
    try:
        transformer = L2Transformer(
            levels=config['pre_processing']['etl']['levels'],
            sampling_ms=config['pre_processing']['etl']['sampling_ms'],
            etl_cfg=config['pre_processing']['etl']
        )
        
        def _process_line(line, rows_dict):
            try:
                msg = json.loads(line)
                res = transformer.process_message(msg)
                if res:
                    if not rows_dict:
                        for k in res.keys(): rows_dict[k] = []
                    for k, v in res.items():
                        rows_dict[k].append(v)
            except: pass

        if l2_file_path.exists():
            # FAST PATH: Unzipped file exists
            with open(l2_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    _process_line(line, rows)
        else:
            # FALLBACK PATH: Process directly from ZIP
            if not zip_path.exists():
                return {"file": zip_path.name, "status": "missing_zip_source"}
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.data'):
                        with z.open(name) as f:
                            for line in f:
                                _process_line(line, rows)
        
        if not rows:
            return {"file": zip_path.name, "status": "no_data"}
            
        df_l2 = pl.DataFrame(rows).sort("ts")
        # Clear rows memory
        del rows
        gc.collect()

        # 2. Load Trades
        df_trades = pl.read_csv(trade_path)
        
        df_trades = df_trades.with_columns([
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64).alias("amount"),
            pl.when(pl.col("side") == "Buy").then(1).otherwise(-1).alias("side")
        ]).sort("ts")
        
        # 3. Join Asof (Backward) - Much more memory efficient than Full Join
        df_combined = df_trades.join_asof(
            df_l2,
            on="ts",
            strategy="backward"
        ).with_columns([
            pl.all().forward_fill().backward_fill()
        ])
        
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

def setup_cloud_directories(config):
    """Cria e LIMPA as pastas de dados brutos para o backtest na Cloud."""
    raw_zip = PROJECT_ROOT / config['pipeline_paths']['raw_zip_dir']
    raw_trades = PROJECT_ROOT / config['pipeline_paths']['raw_trades_dir']
    
    for folder in [raw_zip, raw_trades]:
        if folder.exists():
            logger.info(f"🧹 Cleaning folder: {folder}")
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

def sync_drive_data(config):
    """Sincroniza os arquivos de Backtest do Drive para a Cloud via rclone."""
    if "/workspace" not in str(PROJECT_ROOT):
        logger.info("🏠 Local environment detected. Skipping rclone sync.")
        return

    rclone_cfg = PROJECT_ROOT / "rclone.conf"
    rclone_bin = "rclone"
    if os.name == 'nt' and (PROJECT_ROOT / "rclone.exe").exists():
        rclone_bin = str((PROJECT_ROOT / "rclone.exe").absolute())

    # Window filter: March 14 to March 26
    # Note: rclone uses glob, {14..26} is bash-only. We use multiple includes or a pattern.
    window_pattern = "*2026-03-{14,15,16,17,18,19,20,21,22,23,24,25,26}*"
    
    sync_jobs = [
        {
            "remote": "drive:PROJETOS/BACKTEST/BTC_USDT_L2_2023_2026/btcusdt_L2_2026",
            "local": PROJECT_ROOT / config['pipeline_paths']['raw_zip_dir']
        },
        {
            "remote": "drive:PROJETOS/BACKTEST/BTC_USDT_L2_TRADE_2023_2026/btcusdt_L2_trade_2026",
            "local": PROJECT_ROOT / config['pipeline_paths']['raw_trades_dir']
        }
    ]

    logger.info(f"🚀 Starting Rclone Sync for window 14-26 March...")
    for job in sync_jobs:
        cmd = [
            rclone_bin, "copy", job["remote"], str(job["local"]),
            "-P", "--include", window_pattern,
            "--transfers", "16", "--checkers", "16"
        ]
        if rclone_cfg.exists():
            cmd += ["--config", str(rclone_cfg)]
        
        logger.info(f"📥 Syncing: {job['remote']} -> {job['local']}")
        subprocess.run(cmd, check=True)

def run_backtest_pipeline_robust(config):
    # 0. Setup and Sync (Cloud Only)
    setup_cloud_directories(config)
    sync_drive_data(config)

    raw_trade_dir = PROJECT_ROOT / config['pipeline_paths']['raw_trades_dir']
    pre_processed_dir = PROJECT_ROOT / config['pipeline_paths']['pre_processed_dir']
    labelled_dir = PROJECT_ROOT / config['pipeline_paths']['labelled_dir']
    raw_zip_dir = PROJECT_ROOT / config['pipeline_paths']['raw_zip_dir']
    
    logger.info(f"📁 Looking for ZIP files in: {raw_zip_dir}")
    if not raw_zip_dir.exists():
        logger.error(f"❌ Directory NOT FOUND: {raw_zip_dir}")
        
    pre_processed_dir.mkdir(parents=True, exist_ok=True)
    labelled_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = sorted(list(raw_zip_dir.glob("*.zip")))
    
    # Dynamic logic: Use config value if exists, else fallback
    max_workers = config.get('processing', {}).get('n_workers', 4)
    
    # ── 1. Parallel ETL ──
    logger.info(f"🚀 Starting PARALLEL Backtest ETL for {len(zip_files)} days with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_zip = {executor.submit(process_single_day_etl, zp, raw_trade_dir, pre_processed_dir, config): zp for zp in zip_files}
        for future in tqdm(as_completed(future_to_zip), total=len(zip_files), desc="ETL Work"):
            res = future.result()
            if res['status'] == 'error':
                logger.error(f"❌ {res['file']}: {res.get('message', '')}")

    # ── 2. Parallel Labelling ──
    pre_files = sorted(list(pre_processed_dir.glob("*.parquet")))
    logger.info(f"🚀 Starting PARALLEL Backtest Labelling for {len(pre_files)} files with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pf = {executor.submit(process_single_file_labelling, pf, config, labelled_dir): pf for pf in pre_files}
        for future in tqdm(as_completed(future_to_pf), total=len(pre_files), desc="Labelling Work"):
            res = future.result()
            if "error" in res:
                logger.error(f"❌ Error labelling {res.get('file')}: {res['error']}")

if __name__ == "__main__":
    config = load_backtest_config()
    run_backtest_pipeline_robust(config)
    logger.info("✅ Integrated Backtest Data Pipeline Complete.")
