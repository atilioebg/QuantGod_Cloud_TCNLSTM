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
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parents[2].absolute()
if str(PROJECT_ROOT) not in sys.path:
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

def _build_quality_summary(quality_audits: list, skipped_files: list, resample_min: int = 5) -> dict:
    """1:1 Mirror of run_pipeline.py logic for summary generation."""
    GAP_BUCKETS = [1, 2, 3, 5, 10, 15, 30, 60]
    total = len(quality_audits)
    clean, with_clip, with_gap, healed, hard_reset, multi_island, invalid = 0, 0, 0, 0, 0, 0, 0
    total_cells_clipped = 0
    feature_clip_totals = {}
    gap_bucket_counts = {str(b): 0 for b in GAP_BUCKETS}
    gap_bucket_counts["60+"] = 0
    total_islands_generated, total_islands_survived, all_row_counts = 0, 0, []

    for audit in quality_audits:
        h = audit.get("health_stats", {})
        is_valid = h.get("is_valid", audit.get("is_valid", False))
        is_healed = audit.get("healed", False)
        n_islands = audit.get("num_islands_generated", 1)
        valid_ids = h.get("valid_island_ids", [])
        n_survived = len(valid_ids) if valid_ids else (1 if is_valid else 0)
        gap_before = float(audit.get("max_gap_before", 0.0))
        clipped = int(audit.get("clipped_count", 0))
        rows = int(audit.get("total_rows_retained", 0))
        per_feat = audit.get("clipped_per_feature", {})

        if not is_valid: invalid += 1
        if clipped > 0:
            with_clip += 1
            total_cells_clipped += clipped
            for feat, cnt in per_feat.items():
                feature_clip_totals[feat] = feature_clip_totals.get(feat, 0) + cnt
        if gap_before > resample_min:
            with_gap += 1
            placed = False
            for b in GAP_BUCKETS:
                if gap_before <= b:
                    gap_bucket_counts[str(b)] += 1
                    placed = True
                    break
            if not placed: gap_bucket_counts["60+"] += 1
        if is_healed: healed += 1
        if n_islands > 1: hard_reset += 1
        if n_survived >= 2: multi_island += 1
        total_islands_generated += n_islands
        total_islands_survived += n_survived
        if rows > 0: all_row_counts.append(rows)
        if clipped == 0 and gap_before <= resample_min and not is_healed and n_islands <= 1 and is_valid:
            clean += 1

    top_features = sorted(feature_clip_totals.items(), key=lambda x: -x[1])[:5]
    row_stats = {"total_rows_retained": int(sum(all_row_counts)), "avg_rows_per_file": round(sum(all_row_counts)/len(all_row_counts), 1) if all_row_counts else 0}
    return {
        "total_files_processed": total, "total_files_clean": clean, "total_files_with_clip": with_clip,
        "total_files_with_gap": with_gap, "total_files_healed": healed, "total_files_invalid": invalid,
        "clipping_stats": {"total_cells_clipped": total_cells_clipped, "top_clipped_features": [{"feature": f, "clip_events": c} for f, c in top_features]},
        "island_stats": {"total_islands_generated": total_islands_generated, "total_islands_survived": total_islands_survived},
        "row_stats": row_stats
    }

def process_single_day_etl(zip_path, raw_trade_dir, pre_processed_dir, config):
    """Refactored Worker Function for total parity with run_pipeline.py."""
    import psutil
    pid = os.getpid()
    peak_vm_used = 0
    
    date_str = zip_path.name.split("_")[0]
    trade_path = raw_trade_dir / f"BTCUSDT{date_str}.csv.gz"
    out_name = zip_path.stem.replace(".data", "") + ".parquet"
    out_path = pre_processed_dir / out_name
    
    try:
        etl_cfg = config['pre_processing']['etl']
        extractor = DataExtractor(str(zip_path.parent), temp_dir=PROJECT_ROOT / "tmp" / "backtest_raw")
        transformer = L2Transformer(levels=etl_cfg['levels'], sampling_ms=etl_cfg['sampling_ms'], etl_cfg=etl_cfg)
        event_sampler = EventSampler(etl_cfg)
        loader = DataLoader(pre_processed_dir)
        validator = DataValidator()

        # ── 1. Trade Scan (Parity) ─────────────────────────────────────
        lf_trades = pl.scan_csv(trade_path)
        trades_schema = lf_trades.collect_schema().names()
        if "timestamp" in trades_schema:
            lf_trades = lf_trades.with_columns((pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts"))
        lf_trades = lf_trades.sort("ts")
        
        # ── 2. L2 Stream (Parity) ───────────────────────────────────────
        transformer.reset_book()
        transformer.audit_report["file_id"] = zip_path.name
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
        
        # ── 3. Production Join (Parity) ──────────────────────────────────
        lf_merged = lf_trades.join_asof(lf_l2, on="ts", strategy="backward").with_columns([
            pl.all().forward_fill().backward_fill().fill_null(0.0)
        ])
        
        merged_schema = lf_merged.collect_schema().names()
        if "price" in merged_schema and "size" in merged_schema:
            lf_merged = lf_merged.with_columns((pl.col("price") * pl.col("size")).alias("usd_volume"))
            
        if "l2_ts" in merged_schema:
            lf_merged = lf_merged.with_columns((pl.col("ts") - pl.col("l2_ts") > 2000).fill_null(True).alias("__stale_l2__")).drop("l2_ts")
            
        df_merged = lf_merged.collect(engine="streaming")
        del lf_trades, lf_l2, lf_merged
        
        vm_peak = psutil.Process(pid).memory_info().rss
        peak_vm_used = max(peak_vm_used, vm_peak)

        # ── 4. Event Sampling & Feature Engineering (Parity) ──────────
        df_bars = event_sampler.compute_event_bars(df_merged)
        del df_merged
        if len(df_bars) == 0: return {"file": zip_path.name, "status": "skipped", "reason": "No bars"}
            
        df_final = event_sampler.apply_feature_engineering_bars(df_bars)
        del df_bars
        
        # ── 5. Auditor Context Features (Integrated for Backtest) ───────
        resample_freq = etl_cfg.get('resample_freq', '5min')
        resample_min = int(pd.to_timedelta(resample_freq).total_seconds() // 60)
        df_final = calculate_context_features_polars(df_final.lazy(), resample_min=resample_min).collect()
        
        # ── 6. Production Validation (Parity) ───────────────────────────
        df_final = df_final.fill_nan(0.0).fill_null(0.0)
        feature_list = config['model'].get('feature_names', [])
        health = validator.validate_integrity(
            df_final, name=date_str, feature_list=feature_list,
            resample_freq=resample_freq, delta_short_min=etl_cfg.get('delta_short_min', 5)
        )
        
        valid_ids = health.get('valid_island_ids', [])
        if valid_ids and 'island_id' in df_final.columns:
            df_final = df_final.filter(pl.col('island_id').is_in(valid_ids))

        transformer.audit_report["num_islands_generated"] = event_sampler.audit_report["islands"]
        transformer.audit_report["total_rows_retained"] = len(df_final)
        transformer.audit_report["health_stats"] = health

        is_valid = bool(health.get('is_valid', False))
        if is_valid and not df_final.is_empty():
            df_final.write_parquet(out_path)
            status = "ok"
        else:
            status = "error"
            
        del df_final
        gc.collect()
        
        return {
            "file": zip_path.name, "status": status, "audit": transformer.audit_report,
            "peak_ram_gb": peak_vm_used / (1024**3)
        }
    except Exception as e:
        return {"file": zip_path.name, "status": "error", "message": str(e)}

def sync_drive_data(config):
    if "/workspace" not in str(PROJECT_ROOT): return
    rclone_bin = "rclone"
    if os.name == 'nt' and (PROJECT_ROOT / "rclone.exe").exists(): rclone_bin = str((PROJECT_ROOT / "rclone.exe").absolute())
    
    window_pattern = "*2026-03-{14,15,16,17,18,19,20,21,22,23,24,25,26}*"
    sync_jobs = [
        {"remote": "drive:PROJETOS/BACKTEST/BTC_USDT_L2_2023_2026/btcusdt_L2_2026", "local": PROJECT_ROOT / config['pipeline_paths']['raw_zip_dir']},
        {"remote": "drive:PROJETOS/BACKTEST/BTC_USDT_L2_TRADE_2023_2026/btcusdt_L2_trade_2026", "local": PROJECT_ROOT / config['pipeline_paths']['raw_trades_dir']}
    ]

    rclone_cfg = PROJECT_ROOT / "rclone.conf"
    for job in sync_jobs:
        job["local"].mkdir(parents=True, exist_ok=True)
        cmd = [rclone_bin, "copy", job["remote"], str(job["local"]), "-P", "--include", window_pattern, "--transfers", "16"]
        if rclone_cfg.exists():
            cmd += ["--config", str(rclone_cfg)]
        subprocess.run(cmd, check=True)

def run_backtest_pipeline_robust(config):
    # 0. Setup and Sync
    # if "/workspace" in str(PROJECT_ROOT):
    #     sync_drive_data(config)

    raw_trade_dir = PROJECT_ROOT / config['pipeline_paths']['raw_trades_dir']
    pre_processed_dir = PROJECT_ROOT / config['pipeline_paths']['pre_processed_dir']
    labelled_dir = PROJECT_ROOT / config['pipeline_paths']['labelled_dir']
    raw_zip_dir = PROJECT_ROOT / config['pipeline_paths']['raw_zip_dir']
    
    pre_processed_dir.mkdir(parents=True, exist_ok=True)
    labelled_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = sorted(list(raw_zip_dir.glob("*.zip")))
    
    # Scale Guard (Parity)
    etl_cfg = config['pre_processing']['etl']
    max_workers = etl_cfg.get('max_workers', 4)
    scale_cfg = etl_cfg.get('scale_guard', {})
    if scale_cfg.get('enabled', False):
        multiplier = scale_cfg.get('thresholds', {}).get(2026, 1.0)
        max_workers = max(1, int(max_workers * multiplier))
        logger.info(f"Scale Guard active: Using {max_workers} processes.")

    # 1. Parallel ETL
    quality_audits = []
    logger.info(f"🚀 Starting PARALLEL Backtest ETL for {len(zip_files)} days...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_zip = {executor.submit(process_single_day_etl, zp, raw_trade_dir, pre_processed_dir, config): zp for zp in zip_files}
        for future in tqdm(as_completed(future_to_zip), total=len(zip_files), desc="ETL Work"):
            res = future.result()
            if res.get('audit'): quality_audits.append(res['audit'])
            if res['status'] == 'error': logger.error(f"❌ {res.get('file')}: {res.get('message', 'Production Health Check Failed')}")

    # 2. Quality Report (Parity)
    if quality_audits:
        report_dir = PROJECT_ROOT / "docs" / "reports" / "backtest"
        report_dir.mkdir(parents=True, exist_ok=True)
        resample_freq = etl_cfg.get('resample_freq', '5min')
        res_min = int(pd.to_timedelta(resample_freq).total_seconds() // 60)
        summary = _build_quality_summary(quality_audits, [], resample_min=res_min)
        with open(report_dir / "data_quality_report.json", "w") as f: json.dump(summary, f, indent=4)
        logger.info(f"📊 Quality Report saved to {report_dir / 'data_quality_report.json'}")

    # 3. Parallel Labelling (Parity)
    pre_files = sorted(list(pre_processed_dir.glob("*.parquet")))
    logger.info(f"🚀 Starting PARALLEL Backtest Labelling for {len(pre_files)} files...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pf = {executor.submit(process_single_file_labelling, pf, config, labelled_dir): pf for pf in pre_files}
        for future in tqdm(as_completed(future_to_pf), total=len(pre_files), desc="Labelling Work"):
            res = future.result()
            if "error" in res: logger.error(f"❌ Error labelling {res.get('file')}: {res['error']}")

if __name__ == "__main__":
    config = load_backtest_config()
    run_backtest_pipeline_robust(config)
    logger.info("✅ Integrated Backtest Data Pipeline Complete (100% Parity).")
