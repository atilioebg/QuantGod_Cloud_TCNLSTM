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

from src.cloud.base_model.pre_processamento.etl.extract import DataExtractor
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.event_sampler import EventSampler
from src.cloud.base_model.pre_processamento.etl.load import DataLoader
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator
from src.cloud.base_model.labelling.run_labelling import process_single_file_labelling
from src.cloud.auditor_model.auditor_preprocessing import calculate_context_features_polars

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
    import psutil
    pid = os.getpid()
    
    date_str = zip_path.name.split("_")[0]
    trade_path = raw_trade_dir / f"BTCUSDT{date_str}.csv.gz"
    out_name = zip_path.stem.replace(".data", "") + ".parquet"
    out_path = pre_processed_dir / out_name
    
    # 1:1 Mirror of run_pipeline.py logic
    try:
        etl_cfg = config['pre_processing']['etl']
        # Local dir as we sync first
        local_l2_dir = PROJECT_ROOT / config['pipeline_paths']['raw_zip_dir']
        extractor = DataExtractor(str(local_l2_dir), temp_dir=PROJECT_ROOT / "tmp" / "backtest_raw")
        transformer = L2Transformer(levels=etl_cfg['levels'], sampling_ms=etl_cfg['sampling_ms'], etl_cfg=etl_cfg)
        event_sampler = EventSampler(etl_cfg)
        loader = DataLoader(pre_processed_dir)
        validator = DataValidator()

        # ── 1. Trade Scan (Mirror production) ───────────────────────────
        lf_trades = pl.scan_csv(trade_path)
        trades_schema = lf_trades.collect_schema().names()
        if "timestamp" in trades_schema:
            lf_trades = lf_trades.with_columns((pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts"))
        lf_trades = lf_trades.sort("ts")
        
        # ── 2. L2 Stream (Mirror production) ─────────────────────────────
        transformer.reset_book()
        sampled_rows = {}
        for name, file_obj in extractor.stream_zip_content(zip_path):
            for line in file_obj:
                if not line: continue
                msg = json.loads(line)
                row = transformer.process_message(msg)
                if row:
                    if not sampled_rows: sampled_rows = {k: [] for k in row.keys()}
                    for k in sampled_rows.keys(): sampled_rows[k].append(row.get(k, 0.0))
        
        if not sampled_rows:
            return {"file": zip_path.name, "status": "no_data"}
            
        df_l2 = pl.DataFrame(sampled_rows)
        lf_l2 = df_l2.lazy().sort("ts").with_columns(pl.col("ts").alias("l2_ts"))
        del sampled_rows
        
        # ── 3. Production Join (Atomic AsOf) ── [run_pipeline.py line 238] ──
        lf_merged = lf_trades.join_asof(
            lf_l2,
            on="ts",
            strategy="backward"
        ).with_columns([
            pl.all().forward_fill().backward_fill().fill_null(0.0) # [Robust uniform fill]
        ])
        
        merged_schema = lf_merged.collect_schema().names()
        if "price" in merged_schema and "size" in merged_schema:
            lf_merged = lf_merged.with_columns((pl.col("price") * pl.col("size")).alias("usd_volume"))
            
        if "l2_ts" in merged_schema:
            lf_merged = lf_merged.with_columns((pl.col("ts") - pl.col("l2_ts") > 2000).fill_null(True).alias("__stale_l2__")).drop("l2_ts")
            
        df_merged = lf_merged.collect(engine="streaming")
        del lf_trades, lf_l2, lf_merged
        
        # ── 4. Event Sampling & Feature Engineering ─────────────────────
        df_bars = event_sampler.compute_event_bars(df_merged)
        del df_merged
        if len(df_bars) == 0: return {"file": zip_path.name, "status": "skipped", "reason": "No bars"}
            
        df_final = event_sampler.apply_feature_engineering_bars(df_bars)
        del df_bars
        
        # ── 5. Auditor Context Features (Alpha Sensors) ──────────────────
        # Fix: The calculation function now handles missing columns internally
        resample_freq = etl_cfg.get('resample_freq', '5min')
        resample_min = int(resample_freq.replace('min', '').replace('m', '').replace('T', ''))
        
        # Diagnostic: Check for NaNs before Alpha Sensors
        nan_before = sum(df_final.null_count().row(0))
        if nan_before > 0:
            logger.warning(f"⚠️  [PRE-AUDIT] Found {nan_before} NaNs. Applying safety fill.")
            df_final = df_final.fill_null(0.0)

        df_final = calculate_context_features_polars(df_final.lazy(), resample_min=resample_min).collect()
        
        # Diagnostic: Check for NaNs after Alpha Sensors
        nan_after = sum(df_final.null_count().row(0))
        if nan_after > 0:
            nan_cols = [c for c in df_final.columns if df_final[c].null_count() > 0]
            logger.warning(f"⚠️  [POST-AUDIT] Found {nan_after} NaNs in columns: {nan_cols}. Cleaning...")
            df_final = df_final.fill_null(0.0)
        
        # ── 6. Production Validation ── [run_pipeline.py line 292] ──────
        feature_list = config['model'].get('feature_names', [])
        health = validator.validate_integrity(
            df_final, 
            name=date_str, 
            feature_list=feature_list,
            resample_freq=etl_cfg.get('resample_freq', '5min')
        )
        
        valid_ids = health.get('valid_island_ids', [])
        if valid_ids and 'island_id' in df_final.columns:
            df_final = df_final.filter(pl.col('island_id').is_in(valid_ids))

        if not bool(health.get('is_valid', False)) or df_final.is_empty():
            return {"file": zip_path.name, "status": "error", "message": "Production Health Check Failed"}

        df_final.write_parquet(out_path)
        del df_final
        gc.collect()
        
        vm = psutil.Process(pid).memory_info().rss / (1024**3)
        logger.info(f"✅ [MIRROR] {out_name} | PID {pid} RAM ~{vm:.2f}GB")
        
        return {"file": zip_path.name, "status": "ok"}
    except Exception as e:
        return {"file": zip_path.name, "status": "error", "message": str(e)}

def setup_cloud_directories(config):
    """Cria e LIMPA as pastas de dados brutos para o backtest na Cloud."""
    raw_zip = PROJECT_ROOT / config['pipeline_paths']['raw_zip_dir']
    raw_trades = PROJECT_ROOT / config['pipeline_paths']['raw_trades_dir']
    
    # Only clean if in Cloud environment to avoid deleting local test data
    if "/workspace" not in str(PROJECT_ROOT):
        logger.info("🏠 Local environment: skipping directory cleaning.")
        return

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
    window_pattern = "*2026-03-{13,14,15,16,17,18,19,20,21,22,23,24,25,26}*"
    
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

    logger.info(f"🚀 Starting Rclone Sync for window 13-26 March...")
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
    
    # CPU Detection logic (v8.0)
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
        
    etl_cfg = config['pre_processing']['etl']
    max_workers = etl_cfg.get('max_workers', 4)
    worker_mode = "Manual"

    # Scale Guard: Adaptive reduction based on configurable thresholds
    scale_cfg = etl_cfg.get('scale_guard', {})
    if scale_cfg.get('enabled', False):
        # We assume 2026 for this backtest dataset
        multiplier = scale_cfg.get('thresholds', {}).get(2026, 1.0)
        if multiplier < 1.0:
            original_workers = max_workers
            max_workers = max(1, int(max_workers * multiplier))
            worker_mode += f" + ScaleGuard ({multiplier}x for 2026)"
            logger.info(f"Scale Guard active: {original_workers} -> {max_workers} workers.")

    logger.info(f"ETL Worker Mode: {worker_mode} -> Using {max_workers} processes.")
    
    # ── 1. Parallel ETL ──
    logger.info(f"🚀 Starting PARALLEL Backtest ETL for {len(zip_files)} days with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_zip = {executor.submit(process_single_day_etl, zp, raw_trade_dir, pre_processed_dir, config): zp for zp in zip_files}
        for future in tqdm(as_completed(future_to_zip), total=len(zip_files), desc="ETL Work"):
            res = future.result()
            if res['status'] == 'error':
                logger.error(f"❌ {res.get('file')}: {res.get('message', '')}")
            elif res['status'] == 'skipped' and "reason" in res:
                logger.warning(f"⚠️ {res['file']} skipped: {res['reason']}")

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
