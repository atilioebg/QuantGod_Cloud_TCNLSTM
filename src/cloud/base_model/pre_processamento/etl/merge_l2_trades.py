import polars as pl
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_l2_trades(l2_path: str, trades_path: str, output_path: str):
    """
    Merges L2 Orderbook data with Trade execution data using as-of join.
    Ensures zero leakage by using 'backward' strategy.
    
    L2 (Parquet): Context (Right side)
    Trades (CSV): Trigger (Left side)
    """
    logger.info(f"Loading L2 data from {l2_path}...")
    # L2 usually has 'ts' in ms
    df_l2 = pl.read_parquet(l2_path)
    
    logger.info(f"Loading Trades data from {trades_path}...")
    # Trades CSV from Bybit usually has 'timestamp' in float seconds (e.g. 1767225600.6194)
    # Let's verify the columns first
    df_trades = pl.read_csv(trades_path)
    
    # 1. Prepare Timestamps (Convert both to ms for integer join)
    if "timestamp" in df_trades.columns:
        # Convert float seconds to integer milliseconds
        df_trades = df_trades.with_columns(
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts")
        )
    
    # Ensure both are sorted by 'ts' (mandatory for join_asof)
    df_trades = df_trades.sort("ts")
    df_l2 = df_l2.sort("ts")
    
    # Track original L2 timestamp to compute staleness later
    df_l2 = df_l2.with_columns(pl.col("ts").alias("l2_ts"))
    
    logger.info("Performing join_asof (strategy='backward') to prevent leakage...")
    # Join: Each trade gets the L2 state IMMEDIATELY PRECEDING it
    # This prevents the model from seeing the price movement caused by the trade itself.
    df_merged = df_trades.join_asof(
        df_l2,
        on="ts",
        strategy="backward"
    )
    
    # 2. Cleanup and Initial Feature Calc
    # Compute dollar value per trade for Dollar Bars
    if "price" in df_merged.columns and "size" in df_merged.columns:
        df_merged = df_merged.with_columns(
            (pl.col("price") * pl.col("size")).alias("usd_volume")
        )
        
    # Mark as stale if L2 snapshot is more than 2000ms old relative to the trade
    if "l2_ts" in df_merged.columns:
        df_merged = df_merged.with_columns(
            (pl.col("ts") - pl.col("l2_ts") > 2000).fill_null(True).alias("__stale_l2__")
        ).drop("l2_ts")
    
    logger.info(f"Saving merged dataset to {output_path} (Rows: {len(df_merged)})")
    df_merged.write_parquet(output_path)
    logger.info("Done.")

if __name__ == "__main__":
    L2_FILE = "2026-01-01_BTCUSDT_raw_table.parquet"
    TRADES_FILE = "BTCUSDT2026-01-01.csv"
    OUTPUT_FILE = "2026-01-01_BTCUSDT_merged_l2_trades.parquet"
    
    if Path(L2_FILE).exists() and Path(TRADES_FILE).exists():
        merge_l2_trades(L2_FILE, TRADES_FILE, OUTPUT_FILE)
    else:
        logger.error(f"Required files not found: {L2_FILE} or {TRADES_FILE}")
