import polars as pl
import yaml
import logging
from pathlib import Path
from src.cloud.base_model.pre_processamento.etl.event_sampler import EventSampler
from src.cloud.base_model.pre_processamento.etl.extract import DataExtractor
from src.cloud.base_model.pre_processamento.etl.transform import L2Transformer
from src.cloud.base_model.labelling.run_labelling import label_triple_barrier
import json
import numpy as np
from tqdm import tqdm
import os
import sys

# Setup basics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_local_subset")

def run_test_on_local_data():
    project_root = Path("c:/Users/Atilio/Desktop/PROJETOS/PESSOAL/QuantGod_Cloud_TCNLSTM")
    # Paths from the user description
    l2_dir = project_root / "PROJETOS/BTC_USDT_L2_2023_2026/btcusdt_L2_2025"
    trade_dir = project_root / "PROJETOS/BTC_USDT_L2_TRADE_2023_2026/btcusdt_L2_trade_2025"
    
    # Target files
    day = "2025-08-20"
    l2_zip = l2_dir / f"{day}_BTCUSDT_ob500.data.zip"
    trade_gz = trade_dir / f"BTCUSDT{day}.csv.gz"
    
    if not l2_zip.exists() or not trade_gz.exists():
        logger.error(f"Files not found: {l2_zip} or {trade_gz}")
        return

    # Load master config
    with open(project_root / "src/cloud/base_model/configs/master_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    etl_cfg = config['pre_processing']['etl']
    transformer = L2Transformer(
        levels=etl_cfg['levels'],
        sampling_ms=etl_cfg['sampling_ms'],
        etl_cfg=etl_cfg
    )
    
    extractor = DataExtractor(path_or_remote="LOCAL", rclone_config=None, temp_dir="tmp/test_subset")
    event_sampler = EventSampler(etl_cfg)
    
    # ── 1. Process L2 ZIP ───────────────────────────────────────────────────
    logger.info(f"Processing L2: {l2_zip.name}...")
    sampled_rows = {}
    for name, file_obj in extractor.stream_zip_content(str(l2_zip)):
        for line in tqdm(file_obj, desc="Processing L2 Messages"):
            if not line: continue
            try:
                msg = json.loads(line)
                row = transformer.process_message(msg)
                if row:
                    if not sampled_rows:
                        sampled_rows = {k: [] for k in row.keys()}
                    for k in sampled_rows.keys():
                        sampled_rows[k].append(row.get(k, np.nan))
            except:
                continue
    
    df_l2 = pl.DataFrame(sampled_rows).sort("ts")
    logger.info(f"L2 Sampled: {len(df_l2)} rows.")
    
    # ── 2. Process Trades ───────────────────────────────────────────────────
    logger.info(f"Processing Trades: {trade_gz.name}...")
    # Polars scan_csv supports gzip
    df_trades = pl.read_csv(str(trade_gz)).with_columns(
        (pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts")
    ).sort("ts")
    logger.info(f"Trades Loaded: {len(df_trades)} rows.")
    
    # ── 3. Merge ─────────────────────────────────────────────────────────────
    logger.info("Merging L2 + Trades (Backward Join)...")
    lf_merged = df_trades.lazy().join_asof(
        df_l2.lazy().with_columns(pl.col("ts").alias("l2_ts")),
        on="ts",
        strategy="backward"
    ).with_columns(
        (pl.col("price") * pl.col("size")).alias("usd_volume"),
        (pl.col("ts") - pl.col("l2_ts") > 2000).fill_null(True).alias("__stale_l2__")
    )
    
    df_merged = lf_merged.collect(streaming=True)
    logger.info(f"Merged Data: {len(df_merged)} rows.")
    
    # ── 4. Event Sampling ───────────────────────────────────────────────────
    logger.info("Running Event Sampler (Dollar/Tick/Info/Time)...")
    df_bars = event_sampler.compute_event_bars(df_merged)
    logger.info(f"Bars Generated: {len(df_bars)} bars.")
    
    df_final = event_sampler.apply_feature_engineering_bars(df_bars)
    # Add island_id if missing (event_sampler should add it, but just in case)
    if "island_id" not in df_final.columns:
        df_final = df_final.with_columns(pl.lit(0).alias("island_id"))
        
    # ── 5. Labelling ────────────────────────────────────────────────────────
    logger.info("Running Triple Barrier Labelling (Refactored v8.2)...")
    # Label requires files, but we can pass a temp file or just modify the function to accept DF
    # Actually run_labelling is designed for files. We can save df_final to parquet temporarily.
    temp_p = Path("tmp/test_subset/pre_processed_test.parquet")
    temp_p.parent.mkdir(parents=True, exist_ok=True)
    df_final = df_final.head(100000)
    df_final.write_parquet(temp_p)
    
    df_labelled = label_triple_barrier([temp_p], config)
    
    logger.info("=== TEST SUCCESSFUL ===")
    logger.info(f"Final Samples (Ready for TCN): {len(df_labelled)}")
    logger.info("Label Distribution:")
    l_counts = df_labelled["target"].value_counts().to_dicts()
    for row in l_counts:
        cls_name = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}.get(row['target'], "UNKNOWN")
        logger.info(f" - {cls_name}: {row['count']} samples")

    # Cleanup
    temp_p.unlink()
    logger.info("Cleaning up temp files...")

if __name__ == "__main__":
    run_test_on_local_data()
