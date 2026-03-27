import polars as pl
import yaml
import logging
import os
import json
from pathlib import Path
from src.cloud.base_model.pre_processamento.etl.extract import DataExtractor
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.pre_processamento.etl.event_sampler import EventSampler
from src.cloud.auditor_model.auditor_preprocessing import calculate_context_features_polars
from src.cloud.base_model.pre_processamento.etl.validate import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock PROJECT_ROOT
PROJECT_ROOT = Path(__name__).absolute().parents[0]

def debug_one_file():
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    etl_cfg = config['pre_processing']['etl']
    sample_l2_zip = Path("data/backtest/L2/raw_zip/2026-03-13_BTCUSDT_ob200.data.zip")
    sample_trades_csv = Path("data/backtest/L2/raw_trades/BTCUSDT2026-03-13.csv.gz")
    
    if not sample_l2_zip.exists() or not sample_trades_csv.exists():
        logger.error("Files not found.")
        return

    extractor = DataExtractor(str(sample_l2_zip.parent), temp_dir=Path("tmp/backtest_raw"))
    transformer = L2Transformer(levels=etl_cfg['levels'], sampling_ms=etl_cfg['sampling_ms'], etl_cfg=etl_cfg)
    event_sampler = EventSampler(etl_cfg)
    validator = DataValidator()
    
    logger.info("Step 1: Streaming L2...")
    transformer.reset_book()
    sampled_rows = {}
    for name, file_obj in extractor.stream_zip_content(sample_l2_zip):
        for line in file_obj:
            if not line: continue
            msg = json.loads(line)
            row = transformer.process_message(msg)
            if row:
                if not sampled_rows: sampled_rows = {k: [] for k in row.keys()}
                for k in sampled_rows.keys(): sampled_rows[k].append(row.get(k, 0.0))
    
    df_l2 = pl.DataFrame(sampled_rows)
    logger.info(f"L2 Sampled: {len(df_l2)} rows")
    
    logger.info("Step 2: Processing Trades...")
    lf_trades = pl.scan_csv(sample_trades_csv)
    if "timestamp" in lf_trades.collect_schema().names():
        lf_trades = lf_trades.with_columns((pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts"))
    
    logger.info("Step 3: Joining...")
    lf_merged = lf_trades.sort("ts").join_asof(
        df_l2.lazy().sort("ts"),
        on="ts",
        strategy="backward"
    ).with_columns([
        pl.all().forward_fill().backward_fill().fill_null(0.0)
    ])
    
    if "price" in lf_merged.collect_schema().names() and "size" in lf_merged.collect_schema().names():
        lf_merged = lf_merged.with_columns((pl.col("price") * pl.col("size")).alias("usd_volume"))

    logger.info("Step 4: Event Sampling & Feature Engineering...")
    df_merged = lf_merged.collect(engine="streaming")
    df_bars = event_sampler.compute_event_bars(df_merged)
    df_final = event_sampler.apply_feature_engineering_bars(df_bars)
    logger.info(f"Features ready: {len(df_final)} rows")
    
    logger.info("Step 5: Auditor Sensors...")
    resample_freq = etl_cfg.get('resample_freq', '5min')
    resample_min = int(resample_freq.replace('min', '').replace('m', '').replace('T', ''))
    df_final = calculate_context_features_polars(df_final.lazy(), resample_min=resample_min).collect()
    logger.info("Auditor sensors ready.")
    
    logger.info("Step 6: Validation...")
    health = validator.validate_integrity(
        df_final, 
        name="2026-03-14", 
        feature_list=config['model']['feature_names']
    )
    logger.info(f"Health Check: {health}")

if __name__ == "__main__":
    debug_one_file()
