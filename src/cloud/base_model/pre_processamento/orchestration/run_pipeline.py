import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from src.cloud.base_model.pre_processamento.etl.extract import DataExtractor
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.load import DataLoader
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator
import json
from tqdm import tqdm
import sys
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.cloud.base_model.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

def process_single_zip(zip_path, config):
    """
    Worker function to process a single ZIP file in parallel.
    """
    try:
        # Initialize modules inside worker for process isolation
        extractor = DataExtractor(
            config['pipeline_paths']['raw_l2_source'],
            rclone_config="rclone.conf"
        )
        transformer = L2Transformer(
            levels=config['pre_processing']['etl']['levels'],
            sampling_ms=config['pre_processing']['etl']['sampling_ms'],
            etl_cfg=config['pre_processing']['etl']
        )
        loader = DataLoader("data/L2/pre_processed_L2")
        validator = DataValidator()

        transformer.reset_book()
        transformer.audit_report["file_id"] = Path(zip_path).name
        # Optimization: use a dictionary of lists instead of a list of dicts
        # This significantly reduces memory overhead and speeds up DataFrame construction
        sampled_rows = {}
        
        # 1. Extraction (Streaming)
        for name, file_obj in extractor.stream_zip_content(zip_path):
            for line in file_obj:
                if not line: continue
                try:
                    msg = json.loads(line)
                    row = transformer.process_message(msg)
                    if row:
                        if not sampled_rows:
                            # Initialize lists for all keys on first successful row
                            sampled_rows = {k: [] for k in row.keys()}
                        
                        # Append values, handle potential missing keys safely
                        for k in sampled_rows.keys():
                            sampled_rows[k].append(row.get(k, np.nan))
                except Exception as e:
                    logger.debug(f"[pipeline] Failed to process message: {e}")
                    continue
        
        # 2. Transformation & Loading
        zip_p = Path(zip_path)
        if sampled_rows:
            df_sampled = pd.DataFrame(sampled_rows)

            if df_sampled.empty:
                logger.warning(f"⚠️  No rows sampled in {zip_p.name} (file might be empty or missing 1min thresholds). Skipping.")
                return f"⚠️  No data in {zip_p.name}"

            df_final = transformer.apply_feature_engineering(df_sampled)
            
            # Architecture Integrity (Gold v4.6): Check if all required features are present
            feature_list = config['model'].get('feature_names', [])
            health_report = validator.validate_integrity(df_final, name=zip_p.name, feature_list=feature_list)
            
            # Merge validator stats into the audit report
            transformer.audit_report["health_stats"] = health_report

            # Save policy: Only save if is_valid is explicitly True
            is_valid = bool(health_report.get('is_valid', False))
            output_name = zip_p.with_suffix(".parquet").name
            
            logger.info(f"AUDIT TRACE: {zip_p.name} -> is_valid={is_valid} (Gap: {health_report.get('max_gap_minutes', 0.0):.1f}min)")
            
            if is_valid:
                saved = loader.save_parquet(df_final, output_name, config['pre_processing']['etl']['export_compression'])
                if saved:
                    logger.info(f"✅ Saved pre-processed data: {output_name}")
            else:
                saved = False
                logger.error(f"❌ REJECTED: {zip_p.name} failed integrity check. Parquet NOT saved to prevent pollution.")
            
            return {
                "status": "success" if saved else "skipped",
                "message": f"✅ Processed {zip_p.name}" if is_valid else f"❌ Rejected {zip_p.name} (Validation Failed)",
                "audit": transformer.audit_report,
                "is_valid": is_valid
            }
        else:
            return {"status": "skipped", "message": f"⚠️  No data in {zip_p.name}"}
            
    except Exception as e:
        zip_name = Path(zip_path).name if zip_path else "unknown"
        return {"status": "error", "message": f"❌ Error processing {zip_name}: {str(e)}"}

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
        rclone_config="rclone.conf"
    )
    # Ensure we start with a clean temp folder
    extractor.cleanup_temp()

    # GOLD CLEANUP: Auto-clean local output folder to prevent rclone from syncing old debris
    local_output = Path("data/L2/pre_processed_L2")
    if local_output.exists():
        logger.info(f"🧹 GOLD CLEANUP: Clearing old parquets in {local_output}")
        for p in local_output.glob("*.parquet"):
            p.unlink()
    else:
        local_output.mkdir(parents=True, exist_ok=True)
    
    zip_files = extractor.list_zips()
    
    if not zip_files:
        logger.error("No data to process.")
        return

    # Dynamic CPU Detection Logic (Consistent with labelling approach)
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
    logger.info(f"Found {len(zip_files)} ZIP files to process.")

    # 4. Parallel Execution with ProcessPool
    skipped_files = []
    failed_files = []
    quality_audits = []
    validation_failures = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future tasks
        future_to_zip = {executor.submit(process_single_zip, zp, config): zp for zp in zip_files}
        
        # Wrap as_completed with tqdm for a beautiful progress bar
        for future in tqdm(as_completed(future_to_zip), total=len(zip_files), desc="Parallel ETL"):
            res_obj = future.result()
            result = res_obj["message"]
            
            if res_obj["status"] == "success":
                quality_audits.append(res_obj["audit"])
                if not res_obj.get("is_valid", True):
                    validation_failures.append(result)
            
            # Optional: Log errors if any
            if res_obj["status"] == "error":
                logger.error(result)
                failed_files.append(result)
            elif res_obj["status"] == "skipped":
                # Track skipped files based on the warning prefix
                file_name = result.split("in ")[-1] if "in " in result else result
                skipped_files.append(file_name)

    # 5. Pipeline Manifest and Auditing
    report_dir = Path("docs/reports")
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
        with open(manifest_path, "w") as f:
            json.dump({
                "total_files_scanned": len(zip_files),
                "total_skipped": len(skipped_files),
                "skipped_files": skipped_files
            }, f, indent=4)
        logger.info(f"📄 Saved skip manifest to {manifest_path} ({len(skipped_files)} files)")

    # 5b. Save Quality Audit Report
    if quality_audits:
        audit_path = report_dir / "data_quality_report.json"
        with open(audit_path, "w") as f:
            json.dump({
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_files": len(quality_audits),
                "clipping_enabled": True,
                "validation_failures_count": len(validation_failures),
                "validation_failures": validation_failures,
                "reports": quality_audits
            }, f, indent=4)
        logger.info(f"📊 Saved data quality report to {audit_path}")
        
        # Check for 10% threshold
        if len(skipped_files) / len(zip_files) > 0.10:
            logger.error(f"⚠️ DATASET SIGNIFICANTLY REDUCED: {len(skipped_files)} files skipped.")
            # Print to standard error/out vigorously in red
            print(f"\033[91m⚠️ DATASET SIGNIFICANTLY REDUCED: {len(skipped_files)} files skipped (>10%). Check docs/reports/pipeline_skip_manifest.json\033[0m")

    logger.info("Pipeline execution finished.")
    logger.info(f"Total processed files: {len(zip_files) - len(skipped_files) - len(failed_files)}")
    logger.info(f"CPUs used: {max_workers} / {cpu_count}")

    # 5c. RUN GOLD STANDARD DATA INTEGRITY TESTS
    try:
        logger.info("🧪 Starting Automated Gold Standard Data Integrity Tests...")
        test_script = "tests/quality/run_quality_tests.py"
        if Path(test_script).exists():
            result = subprocess.run([sys.executable, test_script], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ GOLD STANDARD CERTIFICATION: All integrity tests passed!")
            else:
                logger.error("❌ GOLD STANDARD FAILURE: Integrity tests did not pass.")
                logger.error(f"Test Output: {result.stdout}")
        else:
            logger.warning(f"⚠️ Test script {test_script} not found. Skipping Gold Certification.")
    except Exception as e:
        logger.error(f"⚠️ Gold Standard testing execution failed: {e}")

    # 6. Automated Export to Google Drive (QuantGod Cloud Extension)
    try:
        num_features = config['model'].get('num_features', 30)
        res_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
        
        # Padrão Gold v4.5: PRE_PROCESSED_L2_V4.5_GOLD_1min_30F
        folder_name = f"PRE_PROCESSED_L2_V4.5_GOLD_{res_freq}_{num_features}F"
        local_src = "data/L2/pre_processed_L2"
        remote_dest = f"drive:PROJETOS/{folder_name}"
        rclone_cfg = Path("rclone.conf")
        
        logger.info(f"🚀 Starting automated export to Drive: {folder_name}...")
        
        cmd = ["rclone", "copy", str(local_src), remote_dest, "-P"]
        if rclone_cfg.exists():
            cmd += ["--config", str(rclone_cfg)]
        
        # Using rclone.exe explicitly on Windows if it exists in root
        if os.name == 'nt' and Path("rclone.exe").exists():
            cmd[0] = str(Path("rclone.exe").absolute())

        subprocess.run(cmd, check=True)
        logger.info(f"✅ Export completed successfully: {remote_dest}")
    except Exception as e:
        logger.error(f"❌ Automated export failed: {e}")

if __name__ == "__main__":
    run_pipeline()
