import yaml
import logging
from pathlib import Path
import pandas as pd
from src.cloud.base_model.pre_processamento.etl.extract import DataExtractor
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.load import DataLoader
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator
import json
from tqdm import tqdm
import sys
import os
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
            config['paths']['rclone_mount'],
            rclone_config=config['paths'].get('rclone_config')
        )
        transformer = L2Transformer(
            levels=config['etl']['orderbook_levels'],
            sampling_ms=config['etl']['sampling_interval_ms']
        )
        loader = DataLoader(config['paths']['processed_output'])
        validator = DataValidator()

        transformer.reset_book()
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
                except:
                    continue
        
        # 2. Transformation & Loading
        zip_p = Path(zip_path)
        if sampled_rows:
            df_sampled = pd.DataFrame(sampled_rows)

            if df_sampled.empty:
                logger.warning(f"⚠️  No rows sampled in {zip_p.name} (file might be empty or missing 1min thresholds). Skipping.")
                return f"⚠️  No data in {zip_p.name}"

            df_final = transformer.apply_feature_engineering(df_sampled)
            
            if config['features']['apply_zscore']:
                df_final = transformer.apply_zscore(df_final, config['paths']['scaler_path'])
            
            validator.validate_integrity(df_final, name=zip_p.name)
            
            output_name = zip_p.with_suffix(".parquet").name
            loader.save_parquet(df_final, output_name, config['etl']['compression'])
            
            logger.info(f"✅ Saved pre-processed data: {output_name} in {config['paths']['processed_output']}")
            return f"✅ Processed {zip_p.name}"
        else:
            return f"⚠️  No data in {zip_p.name}"
            
    except Exception as e:
        zip_name = Path(zip_path).name if zip_path else "unknown"
        return f"❌ Error processing {zip_name}: {str(e)}"

def run_pipeline():
    # 1. Load Config
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("src/cloud/base_model/pre_processamento/configs/cloud_config.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Logging
    setup_logger("etl")

    # 3. Setup parallel execution
    extractor = DataExtractor(
        config['paths']['rclone_mount'],
        rclone_config=config['paths'].get('rclone_config')
    )
    # Ensure we start with a clean temp folder
    extractor.cleanup_temp()
    
    zip_files = extractor.list_zips()
    
    if not zip_files:
        logger.error("No data to process.")
        return

    # Dynamic CPU Detection
    # We use CPU count minus 1 to keep the system responsive, with a minimum of 1
    total_cpus = os.cpu_count() or 1
    # max_workers = max(1, total_cpus - 1)
    max_workers = config['etl'].get('max_workers', 4)    
    logger.info(f"System detected {total_cpus} vCPUs. Using {max_workers} parallel workers.")
    logger.info(f"Found {len(zip_files)} ZIP files to process.")

    # 4. Parallel Execution with ProcessPool
    skipped_files = []
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future tasks
        future_to_zip = {executor.submit(process_single_zip, zp, config): zp for zp in zip_files}
        
        # Wrap as_completed with tqdm for a beautiful progress bar
        for future in tqdm(as_completed(future_to_zip), total=len(zip_files), desc="Parallel ETL"):
            result = future.result()
            # Optional: Log errors if any
            if "❌" in result:
                logger.error(result)
                failed_files.append(result)
            elif "⚠️" in result:
                # Track skipped files based on the warning prefix
                file_name = result.split("in ")[-1] if "in " in result else result
                skipped_files.append(file_name)

    # 5. Pipeline Manifest and Auditing
    report_dir = Path("docs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    if skipped_files:
        manifest_path = report_dir / "pipeline_skip_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "total_files_scanned": len(zip_files),
                "total_skipped": len(skipped_files),
                "skipped_files": skipped_files
            }, f, indent=4)
        logger.info(f"📄 Saved skip manifest to {manifest_path} ({len(skipped_files)} files)")
        
        # Check for 10% threshold
        if len(skipped_files) / len(zip_files) > 0.10:
            logger.error(f"⚠️ DATASET SIGNIFICANTLY REDUCED: {len(skipped_files)} files skipped.")
            # Print to standard error/out vigorously in red
            print(f"\033[91m⚠️ DATASET SIGNIFICANTLY REDUCED: {len(skipped_files)} files skipped (>10%). Check docs/reports/pipeline_skip_manifest.json\033[0m")

    logger.info("Pipeline execution finished.")
    logger.info(f"Total processed files: {len(zip_files) - len(skipped_files) - len(failed_files)}")
    logger.info(f"CPUs used: {max_workers} / {total_cpus}")

if __name__ == "__main__":
    run_pipeline()
