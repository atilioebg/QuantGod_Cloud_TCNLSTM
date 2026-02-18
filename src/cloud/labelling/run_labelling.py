import polars as pl
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Logger setup
log_dir = Path("logs/labelling")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "labelling_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def apply_labelling(file_path, config):
    """
    Applies asymmetric labelling logic to a single parquet file.
    """
    try:
        input_dir = Path(config['paths']['input_dir'])
        output_dir = Path(config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        params = config['params']
        lookahead = params['lookahead']
        threshold_long = params['threshold_long']
        threshold_short = params['threshold_short']
        
        # 1. Load Parquet (Selective Load to save RAM)
        # We need all columns + the close column for labelling
        df = pl.read_parquet(file_path)
        
        # 2. Calculate Future Returns
        # We want the cumulative log return from t+1 up to t+lookahead.
        # rolling_sum(60) at index t+60 gives sum(t+1...t+60).
        # shifting that back to index t gives exactly the future 60-min return.
        df = df.with_columns([
            pl.col("log_ret_close").rolling_sum(window_size=lookahead).shift(-lookahead).alias("future_return")
        ])
        
        # 3. Apply Thresholds
        df = df.with_columns([
            pl.when(pl.col("future_return") > threshold_long).then(2) # BUY
            .when(pl.col("future_return") < threshold_short).then(0) # SELL
            .otherwise(1) # NEUTRAL
            .alias("target")
        ])
        
        # 4. Cleanup
        # Remove the lookahead rows at the end (where future_return is NaN)
        df_final = df.slice(0, len(df) - lookahead).drop("future_return")
        
        # 5. Save
        output_path = output_dir / file_path.name
        df_final.write_parquet(output_path)
        
        return {
            "status": "success",
            "file": file_path.name,
            # Converter para dict simples {class: count}
            # value_counts retorna struct com colunas "target" e "count"
            # Precisamos iterar
            "counts": {
                row['target']: row['count'] 
                for row in df_final['target'].value_counts().to_dicts()
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "file": file_path.name,
            "error": str(e)
        }

def run_labelling():
    # 1. Load Config
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("src/cloud/labelling/labelling_config.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. List Files
    input_dir = Path(config['paths']['input_dir'])
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return

    # Dynamic CPU Detection
    total_cpus = os.cpu_count() or 1
    max_workers = max(1, total_cpus - 1)
    
    logger.info(f"System detected {total_cpus} vCPUs. Using {max_workers} parallel workers for labelling.")
    logger.info(f"Found {len(parquet_files)} files to label.")

    # 3. Parallel Execution
    label_counts_total = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(apply_labelling, pf, config): pf for pf in parquet_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(parquet_files), desc="Labelling Progress"):
            result = future.result()
            
            if result['status'] == 'error':
                 logger.error(f"❌ Error labelling {result['file']}: {result['error']}")
            else:
                 # Aggregate counts
                 for label_class, count in result['counts'].items():
                     label_counts_total[label_class] = label_counts_total.get(label_class, 0) + count

    logger.info("Labelling phase finished.")
    logger.info("Final Label Distribution:")
    for label_class, count in sorted(label_counts_total.items()):
        label_name = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}.get(label_class, f"Class {label_class}")
        logger.info(f"   {label_name} ({label_class}): {count:,} samples")

    logger.info(f"Total processed files: {len(parquet_files)}")
    logger.info(f"CPUs used: {max_workers} / {total_cpus}")

if __name__ == "__main__":
    run_labelling()
