import yaml
import logging
from pathlib import Path
import pandas as pd
from src.cloud.pre_processamento.etl.extract import DataExtractor
from src.cloud.pre_processamento.etl.transform import L2Transformer
from src.cloud.pre_processamento.etl.load import DataLoader
from src.cloud.pre_processamento.etl.validate import DataValidator
import json
from tqdm import tqdm
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Logger setup
log_dir = Path("logs/etl")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "etl_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def process_single_zip(zip_path, config):
    """
    Worker function to process a single ZIP file in parallel.
    """
    try:
        # Initialize modules inside worker for process isolation
        extractor = DataExtractor(config['paths']['rclone_mount'])
        transformer = L2Transformer(
            levels=config['etl']['orderbook_levels'],
            sampling_ms=config['etl']['sampling_interval_ms']
        )
        loader = DataLoader(config['paths']['processed_output'])
        validator = DataValidator()

        transformer.reset_book()
        sampled_rows = []
        
        # 1. Extraction (Streaming)
        for name, file_obj in extractor.stream_zip_content(zip_path):
            for line in file_obj:
                if not line: continue
                try:
                    msg = json.loads(line)
                    row = transformer.process_message(msg)
                    if row:
                        sampled_rows.append(row)
                except:
                    continue
        
        # 2. Transformation & Loading
        if sampled_rows:
            df_sampled = pd.DataFrame(sampled_rows)
            df_final = transformer.apply_feature_engineering(df_sampled)
            
            if config['features']['apply_zscore']:
                df_final = transformer.apply_zscore(df_final, config['paths']['scaler_path'])
            
            validator.validate_integrity(df_final, name=zip_path.name)
            
            output_name = zip_path.with_suffix(".parquet").name
            loader.save_parquet(df_final, output_name, config['etl']['compression'])
            return f"✅ Processed {zip_path.name}"
        else:
            return f"⚠️  No data in {zip_path.name}"
            
    except Exception as e:
        return f"❌ Error processing {zip_path.name}: {str(e)}"

def run_pipeline():
    # 1. Load Config
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("src/cloud/pre_processamento/configs/cloud_config.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup parallel execution
    extractor = DataExtractor(config['paths']['rclone_mount'])
    zip_files = extractor.list_zips()
    
    if not zip_files:
        logger.error("No data to process.")
        return

    # Dynamic CPU Detection
    # We use CPU count minus 1 to keep the system responsive, with a minimum of 1
    total_cpus = os.cpu_count() or 1
    max_workers = max(1, total_cpus - 1)
    
    logger.info(f"System detected {total_cpus} vCPUs. Using {max_workers} parallel workers.")
    logger.info(f"Found {len(zip_files)} ZIP files to process.")

    # 3. Parallel Execution with ProcessPool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future tasks
        future_to_zip = {executor.submit(process_single_zip, zp, config): zp for zp in zip_files}
        
        # Wrap as_completed with tqdm for a beautiful progress bar
        for future in tqdm(as_completed(future_to_zip), total=len(zip_files), desc="Parallel ETL"):
            result = future.result()
            # Optional: Log errors if any
            if "❌" in result:
                logger.error(result)

    logger.info("Pipeline execution finished.")
    logger.info(f"Total processed files: {len(zip_files)}")
    logger.info(f"CPUs used: {max_workers} / {total_cpus}")

if __name__ == "__main__":
    run_pipeline()
