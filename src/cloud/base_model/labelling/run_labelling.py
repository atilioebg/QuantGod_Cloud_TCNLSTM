import polars as pl
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import os
import subprocess
import pandas as pd  # For to_timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.cloud.base_model.utils.logging_utils import setup_logger, get_labelling_suffix

logger = logging.getLogger(__name__)

def apply_labelling(file_path, config):
    """
    Applies asymmetric labelling logic to a single parquet file.
    The lookahead window is converted from minutes to BARS based on resample_freq,
    ensuring label correctness regardless of temporal resolution.
    """
    try:
        sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
        buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
        mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)

        # v4.9 Gold: Convert horizon_minutes to BARS (not raw minutes).
        # With resample_freq='5min', horizon=15min → 3 bars (not 15).
        resample_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
        resample_min  = int(pd.to_timedelta(resample_freq).total_seconds() // 60)
        lookahead_bars = max(1, mins // resample_min)

        suffix = f"_labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
        output_dir = Path(f"data/L2/splits{suffix}")
        output_dir.mkdir(parents=True, exist_ok=True)

        threshold_long = buy_th
        threshold_short = -sell_th
        
        # 1. Load Parquet (Selective Load to save RAM)
        df = pl.read_parquet(file_path)
        
        # 2. Future Logic - Sniper v4.6 Gold
        # Lookahead: calculate max high and min low in the next 'lookahead_bars' BARS
        # Zero Tolerance Rule: Invalidate if any bar has tick_count == 0 (market silence)
        # Protocol Island Split: Use 'over' to prevent looking into different islands

        island_col = 'island_id' if 'island_id' in df.columns else None

        if island_col:
            logger.debug(f"[labelling] Partitioning by {island_col} for {file_path.name} | lookahead={lookahead_bars} bars ({mins}min)")
            df = df.with_columns([
                pl.col("high").rolling_max(window_size=lookahead_bars).shift(-lookahead_bars).over(island_col).alias("future_max_high"),
                pl.col("low").rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).over(island_col).alias("future_min_low"),
                pl.col("tick_count").rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).over(island_col).alias("future_min_ticks")
            ])
        else:
            df = df.with_columns([
                pl.col("high").rolling_max(window_size=lookahead_bars).shift(-lookahead_bars).alias("future_max_high"),
                pl.col("low").rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).alias("future_min_low"),
                pl.col("tick_count").rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).alias("future_min_ticks")
            ])
        
        # 3. Apply Thresholds & Zero Tolerance
        df = df.with_columns([
            pl.when(pl.col("future_min_ticks").is_null()).then(None) # Proteção contra fim da ilha
            .when(pl.col("future_min_ticks") == 0).then(None)       # ZERO TOLERANCE: Gap no futuro = Inválido
            .when(pl.col("future_max_high") >= pl.col("close") * (1 + buy_th)).then(2)  # BUY
            .when(pl.col("future_min_low") <= pl.col("close") * (1 - sell_th)).then(0)  # SELL
            .otherwise(1) # NEUTRAL
            .alias("target")
        ])
        
        # 4. Calculation of Invalidation Rate
        total_valid_rows = len(df) - lookahead_bars
        invalidated_rows = df.slice(0, total_valid_rows).filter(pl.col("target").is_null()).height
        invalidation_rate = invalidated_rows / total_valid_rows if total_valid_rows > 0 else 0
        
        # 5. Cleanup
        # Remove the lookahead bars + drop invalid ones (Gold Standard Policy)
        # drop_nulls ensures the model is NOT trained with doubtful futures
        df_final = df.slice(0, total_valid_rows).drop_nulls(subset=["target"]).drop(["future_max_high", "future_min_low", "future_min_ticks"])
        
        # 6. Save
        output_path = output_dir / file_path.name
        df_final.write_parquet(output_path)
        
        return {
            "status": "success",
            "file": file_path.name,
            "invalidation_rate": invalidation_rate,
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
    # 1. Load Config (Base always loaded)
    base_config_path = Path("src/cloud/base_model/configs/master_config.yaml")
    if not base_config_path.exists():
        logger.error(f"Base Config file not found at {base_config_path}")
        return

    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    suffix = f"_labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    setup_logger("labelling", suffix)
    
    # Output path based on pipeline root
    base_output = Path(f"data/L2/splits{suffix}")
    
    # 3. List Files
    input_dir = Path("data/L2/pre_processed_L2")
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return

    # Worker count: controlled by master_config.yaml
    # use_dynamic_workers=true  → os.cpu_count()-1 para máxima portabilidade
    # use_dynamic_workers=false → max_workers do config (útil em cloud com CPUs fixas)
    total_cpus = os.cpu_count() or 1
    use_dynamic = config['pre_processing']['labelling'].get('use_dynamic_workers', False)
    if use_dynamic:
        max_workers = max(1, total_cpus - 1)
    else:
        max_workers = config['pre_processing']['labelling'].get('max_workers', 14)

    logger.info(f"System detected {total_cpus} vCPUs. Using {max_workers} parallel workers for labelling (use_dynamic_workers={use_dynamic}).")
    logger.info(f"Found {len(parquet_files)} files to label.")

    # 3. Parallel Execution
    label_counts_total = {}
    total_invalidation_sum = 0
    valid_file_count = 0

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
                 
                 total_invalidation_sum += result.get('invalidation_rate', 0)
                 valid_file_count += 1

    avg_invalidation = (total_invalidation_sum / valid_file_count * 100) if valid_file_count > 0 else 0

    logger.info("Labelling phase finished.")
    logger.info(f"📊 Sniper Audit: Average Target Invalidation Rate: {avg_invalidation:.2f}%")
    logger.info("Final Label Distribution:")
    for label_class, count in sorted(label_counts_total.items()):
        label_name = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}.get(label_class, f"Class {label_class}")
        logger.info(f"   {label_name} ({label_class}): {count:,} samples")

    logger.info(f"Total processed files: {len(parquet_files)}")
    logger.info(f"CPUs used: {max_workers} / {total_cpus}")

    # 4. Automated Export to Google Drive (QuantGod Cloud Extension)
    try:
        num_features = config['model'].get('num_features', 30)
        res_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
        
        # Padrão Gold v4.5: LABELLED_L2_V4.5_GOLD_1min_30F
        folder_name = f"LABELLED_L2_V4.5_GOLD_{res_freq}_{num_features}F"
        local_src = str(base_output)
        remote_dest = f"drive:PROJETOS/{folder_name}"
        rclone_cfg = Path("rclone.conf")
        
        logger.info(f"🚀 Starting automated export to Drive: {folder_name}...")
        
        # --- Run QA Tests Before Export ---
        logger.info("🧪 Running Automated Health QA (pytest)...")
        qa_log_path = Path(local_src) / "labelling_health_QA.log"
        try:
            # Capture terminal output of pytest directly to the labelling folder
            with open(qa_log_path, 'w', encoding='utf-8') as qa_file:
                subprocess.run(
                    ["pytest", "tests/labelling/test_labelling_output.py", "-v"],
                    stdout=qa_file,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PRE_PROCESSED_DIR=str("data/L2/pre_processed_L2"), LABELLED_DIR=local_src),
                    check=False  # Do not raise exception if tests fail - log it and continue
                )
            logger.info(f"✅ QA Report saved to {qa_log_path}")
        except Exception as e:
            logger.error(f"⚠️ QA Report generation failed: {e}")

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
    run_labelling()
