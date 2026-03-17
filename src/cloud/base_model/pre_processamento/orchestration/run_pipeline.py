import yaml
import logging
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
from src.cloud.base_model.pre_processamento.etl.extract import DataExtractor
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.event_sampler import EventSampler
from src.cloud.base_model.pre_processamento.etl.load import DataLoader
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator
import json
from tqdm import tqdm
import sys
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.cloud.base_model.utils.logging_utils import setup_logger, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import (
    get_pre_processed_dir, get_temp_raw_dir,
    get_reports_root, get_logs_root
)

logger = logging.getLogger(__name__)


def _build_quality_summary(quality_audits: list, skipped_files: list, resample_min: int = 5) -> dict:
    """
    v4.9: Aggregates all per-file audit dicts into a high-level descriptive summary
    for the data_quality_report.json. Runs in O(n) over quality_audits.

    Gap Buckets (minutes): 1, 2, 3, 5, 10, 15, 30, 60, 60+
    """
    GAP_BUCKETS = [1, 2, 3, 5, 10, 15, 30, 60]

    total          = len(quality_audits)
    clean          = 0
    with_clip      = 0
    with_gap       = 0
    healed         = 0
    hard_reset     = 0  # island split activated (>1 island generated)
    multi_island   = 0  # >=2 islands survived
    invalid        = 0

    total_cells_clipped  = 0
    feature_clip_totals  = {}   # feature_name -> total clips across all files
    gap_bucket_counts    = {str(b): 0 for b in GAP_BUCKETS}
    gap_bucket_counts["60+"] = 0

    total_islands_generated = 0
    total_islands_survived  = 0
    all_row_counts          = []

    for audit in quality_audits:
        h = audit.get("health_stats", {})
        is_valid   = h.get("is_valid", audit.get("is_valid", False))
        is_healed  = audit.get("healed", False)
        n_islands  = audit.get("num_islands_generated", 1)
        valid_ids  = audit.get("valid_island_ids", [])
        n_survived = len(valid_ids) if valid_ids else (1 if is_valid else 0)
        gap_before = float(audit.get("max_gap_before", 0.0))
        clipped    = int(audit.get("clipped_count", 0))
        rows       = int(audit.get("total_rows_retained", 0))
        per_feat   = audit.get("clipped_per_feature", {})

        if not is_valid:
            invalid += 1

        if clipped > 0:
            with_clip += 1
            total_cells_clipped += clipped
            for feat, cnt in per_feat.items():
                feature_clip_totals[feat] = feature_clip_totals.get(feat, 0) + cnt

        if gap_before > resample_min:
            with_gap += 1
            # Assign to closest bucket
            placed = False
            for b in GAP_BUCKETS:
                if gap_before <= b:
                    gap_bucket_counts[str(b)] += 1
                    placed = True
                    break
            if not placed:
                gap_bucket_counts["60+"] += 1

        if is_healed:
            healed += 1

        if n_islands > 1:
            hard_reset += 1

        if n_survived >= 2:
            multi_island += 1

        total_islands_generated += n_islands
        total_islands_survived  += n_survived

        if rows > 0:
            all_row_counts.append(rows)

        # Clean = no clip, no gap, no healing required
        if clipped == 0 and gap_before <= resample_min and not is_healed and n_islands <= 1 and is_valid:
            clean += 1

    # Top-5 clipped features (sorted by total events desc)
    top_features = sorted(feature_clip_totals.items(), key=lambda x: -x[1])[:5]
    top_features_list = [{"feature": f, "clip_events": c} for f, c in top_features]

    row_stats = {}
    if all_row_counts:
        row_stats = {
            "total_rows_retained": int(sum(all_row_counts)),
            "avg_rows_per_file":   round(sum(all_row_counts) / len(all_row_counts), 1),
            "min_rows_in_file":    int(min(all_row_counts)),
            "max_rows_in_file":    int(max(all_row_counts)),
        }

    # Remove zero-count gap buckets to keep JSON clean
    gap_dist = {k: v for k, v in gap_bucket_counts.items() if v > 0}

    return {
        "total_files_processed":   total,
        "total_files_clean":       clean,
        "total_files_with_clip":   with_clip,
        "total_files_with_gap":    with_gap,
        "total_files_healed":      healed,
        "total_files_hard_reset":  hard_reset,
        "total_files_multi_island": multi_island,
        "total_files_invalid":     invalid,
        "total_files_skipped":     len(skipped_files),
        "gap_distribution_minutes": gap_dist,
        "clipping_stats": {
            "total_cells_clipped":  total_cells_clipped,
            "top_clipped_features": top_features_list,
        },
        "island_stats": {
            "total_islands_generated": total_islands_generated,
            "total_islands_survived":  total_islands_survived,
            "total_islands_abandoned": total_islands_generated - total_islands_survived,
        },
        "row_stats": row_stats,
    }


def process_single_day(zip_path, csv_path, trades_remote, config):
    """
    Worker function to process a single day (L2 ZIP + Trades CSV) in parallel.
    v8.0 Event-Driven: 
      1. Extrai L2 para Polars DF
      2. Extrai Trades para Polars DF
      3. Merge Backward AsOf (Evita lookahead)
      4. Constroi barras por Evento (Dollar/Tick/Info)
    """
    try:
        extractor = DataExtractor(
            config['pipeline_paths']['raw_l2_source'],
            rclone_config="rclone.conf",
            temp_dir=get_temp_raw_dir(config)
        )
        
        etl_cfg = config['pre_processing']['etl']
        transformer = L2Transformer(
            levels=etl_cfg['levels'],
            sampling_ms=etl_cfg['sampling_ms'],
            etl_cfg=etl_cfg
        )
        
        event_sampler = EventSampler(etl_cfg)
        loader = DataLoader(get_pre_processed_dir(config))
        validator = DataValidator()

        zip_p = Path(zip_path)
        day_identifier = zip_p.name.replace(".zip", "")
        
        # ── 1. Download/Parse Trades CSV ─────────────────────────────────────
        try:
            local_csv_path = extractor.download_file(Path(csv_path).name, trades_remote)
            df_trades = pl.read_csv(local_csv_path)
            if "timestamp" in df_trades.columns:
                df_trades = df_trades.with_columns(
                    (pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts")
                )
            df_trades = df_trades.sort("ts")
        except Exception as e:
            return {"status": "error", "message": f"❌ Error loading Trades for {day_identifier}: {e}", "reason": str(e)}
        
        # ── 2. Download/Parse L2 ZIP ─────────────────────────────────────────
        transformer.reset_book()
        transformer.audit_report["file_id"] = day_identifier
        sampled_rows = {}
        
        for name, file_obj in extractor.stream_zip_content(zip_path):
            for line in file_obj:
                if not line: continue
                try:
                    msg = json.loads(line)
                    row = transformer.process_message(msg)
                    if row:
                        if not sampled_rows:
                            sampled_rows = {k: [] for k in row.keys()}
                        for k in sampled_rows.keys():
                            sampled_rows[k].append(row.get(k, np.nan))
                except Exception as e:
                    logger.debug(f"[pipeline] Failed to process message: {e}")
                    continue
                    
        if not sampled_rows or not any(sampled_rows.values()):
            return {"status": "skipped", "message": f"⚠️  No L2 data in {day_identifier}", "reason": "No rows sampled"}

        df_l2 = pl.DataFrame(sampled_rows)
        if "ts" not in df_l2.columns:
            return {"status": "error", "message": f"❌ L2 TS corrupted for {day_identifier}", "reason": "No TS"}
            
        df_l2 = df_l2.sort("ts").with_columns(pl.col("ts").alias("l2_ts"))
        
        # ── 3. Merge L2 + Trades (Atomic join_asof) ──────────────────────────
        df_merged = df_trades.join_asof(
            df_l2,
            on="ts",
            strategy="backward"
        )
        
        if "price" in df_merged.columns and "size" in df_merged.columns:
            df_merged = df_merged.with_columns(
                (pl.col("price") * pl.col("size")).alias("usd_volume")
            )
            
        if "l2_ts" in df_merged.columns:
            df_merged = df_merged.with_columns(
                (pl.col("ts") - pl.col("l2_ts") > 2000).fill_null(True).alias("__stale_l2__")
            ).drop("l2_ts")
            
        # ── 4. Event Sampling (Dollar/Tick/Info) ─────────────────────────────
        df_bars = event_sampler.compute_event_bars(df_merged)
        
        if len(df_bars) == 0:
            return {"status": "skipped", "message": f"⚠️  No bars generated for {day_identifier}", "reason": "Insufficient volume/ticks"}
            
        df_final = event_sampler.apply_feature_engineering_bars(df_bars)
        
        # Merge QA Audits
        transformer.audit_report["num_islands_generated"] = event_sampler.audit_report["islands"]
        transformer.audit_report["total_rows_retained"] = len(df_final)
        
        # Architecture Integrity Validation
        feature_list = config['model'].get('feature_names', [])
        resample_freq = etl_cfg.get('resample_freq', '5min')
        delta_short_min = etl_cfg.get('delta_short_min', 5)
        health_report = validator.validate_integrity(
            df_final,
            name=day_identifier,
            feature_list=feature_list,
            resample_freq=resample_freq,
            delta_short_min=delta_short_min
        )
        
        # Pruning invalid islands
        valid_ids = health_report.get('valid_island_ids', [])
        if valid_ids and 'island_id' in df_final.columns:
            df_final = df_final.filter(pl.col('island_id').is_in(valid_ids))
            transformer.audit_report["total_rows_retained"] = len(df_final)
            
        transformer.audit_report["health_stats"] = health_report

        # Save Action
        is_valid = bool(health_report.get('is_valid', False))
        output_name = f"{day_identifier}.parquet"
        
        is_empty = df_final.is_empty()
        if is_valid and not is_empty:
            saved = loader.save_parquet(df_final, output_name, etl_cfg['export_compression'])
            if saved:
                logger.info(f"✅ Saved event-driven dataset: {output_name}")
        else:
            saved = False
            reason = "Validation Failed" if not is_valid else "Empty output"
            logger.error(f"❌ REJECTED: {output_name} -> {reason}")
            
        # Cleanup temp CSV
        try:
            local_csv_path.unlink()
        except:
            pass
            
        return {
            "status": "success" if saved else "skipped",
            "message": f"✅ Processed {day_identifier}" if is_valid and not is_empty else f"❌ Rejected {day_identifier}",
            "audit": transformer.audit_report,
            "is_valid": is_valid and not is_empty,
            "reason": health_report.get('integrity_comment', "Validation Failed") if not is_valid else None
        }

    except Exception as e:
        zip_name = Path(zip_path).name if zip_path else "unknown"
        return {"status": "error", "message": f"❌ Error processing {zip_name}: {str(e)}", "reason": str(e)}

def run_pipeline():
    # 1. Load Config
    config_path = Path("src/cloud/base_model/configs/master_config.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. Setup Logging
    setup_logger("etl")

    # 3. Setup parallel execution
    extractor = DataExtractor(
        config['pipeline_paths']['raw_l2_source'],
        rclone_config="rclone.conf",
        temp_dir=get_temp_raw_dir(config)
    )
    # Ensure we start with a clean temp folder
    extractor.cleanup_temp()

    # GOLD CLEANUP: Auto-clean local output folder to prevent rclone from syncing old debris
    # Pasta dinâmica: PRE_PROCESSED_L2_{horizon}_{lookback}_{freq}
    local_output = Path(get_pre_processed_dir(config))
    if local_output.exists():
        logger.info(f"🧹 GOLD CLEANUP: Clearing old parquets in {local_output}")
        for p in local_output.glob("*.parquet"):
            p.unlink()
    else:
        local_output.mkdir(parents=True, exist_ok=True)
    
    zip_files = extractor.list_zips()
    trades_remote = config['pipeline_paths']['raw_trades_source']
    csv_files = extractor.list_trades_csvs(trades_remote)
    
    # Parear arquivos por data (YYYY-MM-DD extraído do nome)
    import re
    def extract_date(filename):
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    zips_dict = {extract_date(Path(z).name): z for z in zip_files if extract_date(Path(z).name)}
    csvs_dict = {extract_date(Path(c).name): c for c in csv_files if extract_date(Path(c).name)}
    
    paired_days = []
    for date, zp in zips_dict.items():
        if date in csvs_dict:
            paired_days.append((zp, csvs_dict[date]))
    
    if not paired_days:
        logger.error("No perfectly paired data (ZIP + CSV) found for any day.")
        return

    # Dynamic CPU Detection Logic
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
        
    etl_cfg = config['pre_processing']['etl']
    use_dynamic = etl_cfg.get('use_dynamic_workers', False)
    
    if use_dynamic:
        max_workers = max(1, cpu_count - 1)
        worker_mode = "Dynamic (vCPUs - 1)"
    else:
        max_workers = etl_cfg.get('max_workers', 4)
        worker_mode = "Manual (Static)"
        
    logger.info(f"System detected {cpu_count} vCPUs allocated.")
    logger.info(f"ETL Worker Mode: {worker_mode} -> Using {max_workers} processes.")
    logger.info(f"Found {len(paired_days)} Paired Days (L2 + Trades) to process.")

    # 4. Parallel Execution with ProcessPool
    skipped_files = []
    failed_files = []
    quality_audits = []
    validation_failures = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_zip = {executor.submit(process_single_day, zp, cp, trades_remote, config): zp for zp, cp in paired_days}
        
        for future in tqdm(as_completed(future_to_zip), total=len(paired_days), desc="Parallel ETL"):
            res_obj = future.result()
            result = res_obj["message"]
            
            if res_obj.get("audit"):
                quality_audits.append(res_obj["audit"])

            if res_obj["status"] == "success":
                if not res_obj.get("is_valid", True):
                    validation_failures.append(result)
            
            # Optional: Log errors if any
            if res_obj["status"] == "error":
                logger.error(result)
                failed_files.append(result)
            elif res_obj["status"] == "skipped":
                # Track skipped files with reason (v4.8)
                file_name = Path(future_to_zip[future]).name
                skipped_files.append({"file": file_name, "reason": res_obj.get("reason", "Unknown")})

    # 5. Pipeline Manifest and Auditing
    report_dir = Path(config['pipeline_paths'].get('local_reports_root', 'docs/reports'))
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 5a. Executive Audit Table (CSV)
    if quality_audits:
        audit_csv_path = report_dir / "audit_summary.csv"
        summary_rows = []
        for audit in quality_audits:
            h = audit.get("health_stats", {})
            
            # Status Logic
            is_valid = h.get("is_valid", False)
            is_healed = audit.get("healed", False)
            
            # Refined Status v4.6
            status = "VALID" if is_valid else "INVALID"
            if is_healed and is_valid:
                status = "FIXED"
                
            # Alert Type Priority
            alert_type = "NONE"
            if h.get("ghost_features"): alert_type = "GHOST_FEATURE"
            elif h.get("dead_features"): alert_type = "DEAD_FEATURE"
            elif h.get("high_tail_count", 0) > 10: alert_type = "HIGH_TAIL_VOL"
            elif h.get("max_gap_minutes", 0) > 25: alert_type = "TIME_GAP"
            
            summary_rows.append({
                "file_name": audit.get("file_id", "unknown"),
                "status": status,
                "num_islands": audit.get("num_islands_generated", 1),
                "rows_retained": audit.get("total_rows_retained", 0),
                "alert_type": alert_type,
                "max_gap_before": f"{audit.get('max_gap_before', 0.0):.2f}",
                "max_gap_after": f"{audit.get('max_gap_after', 0.0):.2f}",
                "clipping_density": f"{audit.get('outlier_density', 0.0):.4f}%",
                "features_healed": ", ".join(audit.get("features_healed", [])),
                "healing_details": " | ".join(audit.get("healing_details", [])),
                "integrity_comment": h.get("integrity_comment", "Passed")
            })
            
        df_summary = pd.DataFrame(summary_rows)
        # Sort by status to show INVALID/FIXED at the top
        if not df_summary.empty:
            df_summary['sort_idx'] = df_summary['status'].map({"INVALID": 0, "FIXED": 1, "VALID": 2})
            df_summary = df_summary.sort_values("sort_idx").drop(columns=['sort_idx'])
            
        df_summary.to_csv(audit_csv_path, index=False)
        logger.info(f"📊 [GOLD] Executive Audit Summary saved to {audit_csv_path}")

    if skipped_files:
        manifest_path = report_dir / "pipeline_skip_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_files_scanned": len(zip_files),
                "total_skipped": len(skipped_files),
                "skipped_files": skipped_files
            }, f, indent=4, ensure_ascii=False)
        logger.info(f"📄 Saved skip manifest to {manifest_path} ({len(skipped_files)} files)")

    # 5b. Save Quality Audit Report
    if quality_audits:
        audit_path = report_dir / "data_quality_report.json"

        # Config metadata for the report header
        etl_cfg_snap = config.get('pre_processing', {}).get('etl', {})
        resample_freq  = etl_cfg_snap.get('resample_freq', '5min')
        resample_min_r = int(pd.to_timedelta(resample_freq).total_seconds() // 60)

        pipeline_cfg_snap = {
            "resample_freq":   resample_freq,
            "levels":          etl_cfg_snap.get('levels', 200),
            "clipping_enabled": etl_cfg_snap.get('clipping', {}).get('enabled', False),
            "p99_multiplier":  etl_cfg_snap.get('clipping', {}).get('p99_multiplier', 10),
            "lookback_minutes": config.get('optimization', {}).get('seq_len', 60) * resample_min_r,
            "flow_depth":       etl_cfg_snap.get('flow_depth', 5),
        }

        summary = _build_quality_summary(quality_audits, skipped_files, resample_min=resample_min_r)

        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp":          pd.Timestamp.now().isoformat(),
                "pipeline_config":    pipeline_cfg_snap,
                "summary":            summary,
                "clipping_enabled":   True,
                "validation_failures_count": len(validation_failures),
                "validation_failures": validation_failures,
                "reports":            quality_audits,
            }, f, indent=4, ensure_ascii=False)
        logger.info(f"📊 Saved enriched data quality report to {audit_path}")
        
        # Check for 10% threshold
        if len(paired_days) > 0 and len(skipped_files) / len(paired_days) > 0.10:
            logger.error(f"⚠️ DATASET SIGNIFICANTLY REDUCED: {len(skipped_files)} files skipped.")
            # Print to standard error/out vigorously in red
            print(f"\033[91m⚠️ DATASET SIGNIFICANTLY REDUCED: {len(skipped_files)} files skipped (>10%). Check docs/reports/pipeline_skip_manifest.json\033[0m")

    logger.info("Pipeline execution finished.")
    logger.info(f"Total processed files: {len(paired_days) - len(skipped_files) - len(failed_files)}")
    logger.info(f"CPUs used: {max_workers} / {cpu_count}")

    # 5c. RUN AUTOMATED DATA INTEGRITY TESTS
    try:
        logger.info("🧪 Starting Automated Data Integrity Tests...")
        test_script = "tests/quality/run_quality_tests.py"
        if Path(test_script).exists():
            result = subprocess.run([sys.executable, test_script], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ INTEGRITY CERTIFICATION: All integrity tests passed!")
            else:
                logger.error("❌ INTEGRITY FAILURE: Integrity tests did not pass.")
                logger.error(f"Test Output: {result.stdout}")
        else:
            logger.warning(f"⚠️ Test script {test_script} not found. Skipping Integrity Certification.")
    except Exception as e:
        logger.error(f"⚠️ Integrity testing execution failed: {e}")

    from src.cloud.base_model.utils.path_utils import get_drive_session_path

    # 6. Automated Export to Google Drive → RESULTADOS_.../PRE_PROCESSED/
    try:
        local_src   = get_pre_processed_dir(config)
        remote_dest = get_drive_session_path("PRE_PROCESSED", config)
        rclone_cfg = Path("rclone.conf")

        logger.info(f"🚀 Starting automated export to Drive: {remote_dest}...")

        # Calculate concurrent transfers based on CPU count (cap at 32 for drive API limits)
        rclone_transfers = str(min(32, cpu_count * 2))

        cmd = ["rclone", "copy", str(local_src), remote_dest, "-P", "--transfers", rclone_transfers, "--checkers", rclone_transfers]
        if rclone_cfg.exists():
            cmd += ["--config", str(rclone_cfg)]

        if os.name == 'nt' and Path("rclone.exe").exists():
            cmd[0] = str(Path("rclone.exe").absolute())

        subprocess.run(cmd, check=True)
        logger.info(f"✅ Export completed successfully: {remote_dest}")
    except Exception as e:
        logger.error(f"❌ Automated export failed: {e}")

    # 7. Audit Reports + Logs → RESULTADOS_.../AUDITORIA/ETL/
    report_root = get_reports_root(config)
    upload_audit_to_drive(
        local_dirs=[f"{get_logs_root(config)}/etl"],
        stage_name="ETL",
        config=config,
        extra_files=[
            f"{report_root}/data_quality_report.json",
            f"{report_root}/audit_summary.csv",
            f"{report_root}/pipeline_skip_manifest.json",
        ],
    )

if __name__ == "__main__":
    run_pipeline()
