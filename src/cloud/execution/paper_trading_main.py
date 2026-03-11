import asyncio
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix path to allow imports from project root
project_root = Path(__file__).parents[3]
if project_root not in sys.path:
    sys.path.append(project_root)

# Internal Imports
from src.cloud.base_model.utils.config_utils import load_config
from src.cloud.execution.exchange_connector import DataBuffer, ExchangeConnector
from src.cloud.execution.streaming_etl import StreamingETL
from src.cloud.execution.inference_service import InferenceService
from src.cloud.execution.virtual_broker import VirtualBroker
from src.cloud.execution.performance_journal import PerformanceJournal

# Setup specialized logger for paper trading
# Log file is FIXED (append mode) so every run accumulates in the same file.
# This means if the system stops and restarts, the full history is preserved and
# the exact same output visible on the terminal is also persisted to disk.
LOG_FILE = Path("logs") / "paper_trading.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')   # 'a' = append across restarts
    ]
)
logger = logging.getLogger("PaperTrading")

# ── Session separator (written once at startup, visible in the persistent log) ──
_SESSION_START = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("=" * 70)
logger.info(f"  🔁 NEW SESSION STARTED AT {_SESSION_START}")
logger.info("=" * 70)

async def main():
    logger.info("🚀 Starting QuantGod Paper Trading System...")
    
    # 1. Configuration & Initialization
    config = load_config()
    symbol = config.get('symbol', 'BTCUSDT').lower()
    
    logger.info("📡 Initializing Data Infrastructure...")
    buffer = DataBuffer()
    connector = ExchangeConnector(symbol, buffer)
    
    logger.info("📊 Initializing Streaming ETL & Scalers...")
    etl = StreamingETL(config)
    
    # Sync settings from ETL (which carefully parses them from config/models)
    resample_min = etl.resample_min
    seq_len = etl.seq_len
    lookahead = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    lookback = seq_len * resample_min
    target = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    
    # --- MODEL INFO BANNER (FULLY DYNAMIC) ---
    lookahead_bars = max(1, lookahead // resample_min)
    logger.info("="*55)
    logger.info("  💎 QUANTGOD SYSTEM CONFIGURATION 💎")
    logger.info(f"  • TIMEFRAME:       {resample_min} min")
    logger.info(f"  • LOOKBACK (PAST):  {lookback} min ({seq_len} bars)")
    logger.info(f"  • LOOKAHEAD (GOAL): {lookahead} min ({lookahead_bars} bars)")
    logger.info(f"  • TARGET PROFIT:    {target*100:.2f}%")
    logger.info(f"  • SEQ_LEN (TENSOR): {seq_len}")
    logger.info("="*55)

    logger.info("🛡️ Initializing Inference Stack (3-Layers)...")
    inference = InferenceService(config)
    
    logger.info("💰 Initializing Virtual Broker...")
    broker = VirtualBroker(initial_balance=10000.0)
    
    logger.info("🗒️ Initializing Performance Journal...")
    journal = PerformanceJournal(
        horizon_minutes=lookahead,
        target_pct=target
    )
    
    # 2. Start Data Ingestion
    logger.info(f"🔗 Connecting to Binance WebSocket for {symbol.upper()}...")
    connector.start()
    
    logger.info(f"📍 Monitoring {symbol.upper()} | System is LIVE and collecting data.")
    logger.info("⏳ Waiting for data buffer to stabilize (2s)...")
    await asyncio.sleep(2)
    
    last_bar_time = None
    
    try:
        while True:
            # Current time in UTC (Standard for Crypto)
            now = datetime.now(timezone.utc)
            ts_ms = int(now.timestamp() * 1000)
            
            # 3. Ingest Data into ETL State
            snapshot = await buffer.get_snapshot()
            current_price = 0.0
            
            if snapshot['depth'] and snapshot['trades']:
                etl.process_data(snapshot['depth'], snapshot['trades'], ts_ms)
                # Use Best Bid/Ask mid as current price for the journal
                # IMPORTANT: Binance API returns prices as strings, must cast to float
                best_bid = float(snapshot['depth'].get('bids', [[0]])[0][0])
                best_ask = float(snapshot['depth'].get('asks', [[0]])[0][0])
                current_price = (best_bid + best_ask) / 2
                
                # Update journal's high/low tracking for all pending predictions
                if current_price > 0:
                    journal.update_market_price(current_price, now)
            
            # 4. Check for Bar Boundary transition (e.g. crossing a 1min mark)
            current_bar_idx = now.minute // resample_min
            if last_bar_time is not None and now.minute != last_bar_time.minute and (now.minute % resample_min == 0):
                logger.info(f"🔔 Bar Close Detected: {now.strftime('%H:%M:%S')} UTC")
                
                # Calculate boundary to filter out incomplete spillover bars
                boundary = now.replace(second=0, microsecond=0)
                
                # Calculate Features
                inputs = etl.on_bar_close(boundary)
                
                if inputs:
                    # 5. Run Full 3-Layer Inference
                    result = inference.predict(
                        foundation_input=inputs['foundation_input'],
                        auditor_context=inputs['auditor_input']
                    )
                    
                    # 6. Virtual Execution
                    price_snap = {
                        "bid": best_bid,
                        "ask": best_ask
                    }
                    
                    broker.execute_signal(result['signal'], price_snap, now)
                    
                    # 7. Record in Performance Journal
                    journal.add_prediction(
                        ts=now,
                        symbol=symbol.upper(),
                        signal=result['signal'],
                        score=result['auditor_score'],
                        current_price=current_price
                    )
                    
                    # 8. Monitoring & Logging
                    stats = broker.get_stats(price_snap['bid'])
                    logger.info(
                        f"📊 [INFERÊNCIA] Sinal: {result['signal']} | "
                        f"Confiança: {result['auditor_score']:.4f} | "
                        f"Equity: ${stats['equity']:.2f} ({stats['pnl_pct']:.2f}%)"
                    )
                    
                    # Detailed Probs log for debugging
                    logger.debug(f"DEBUG: Found_Probs: {result['probs_foundation']} | Spec_Probs: {result['probs_specialist']}")
                
            last_bar_time = now
            
            # Sleep until next check (e.g. 1s resolution for ETL injection)
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Termination requested by user.")
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR in Main Loop: {e}", exc_info=True)
    finally:
        if 'etl' in locals():
            logger.info("💾 Triggering final state save before shutdown...")
            etl.save_state()
        if 'connector' in locals():
            connector.stop()
        logger.info("👋 System shutdown complete.")

if __name__ == "__main__":
    # Create logs dir if not exists
    Path("logs").mkdir(exist_ok=True)

    def _backup_log_to_resultados():
        """Copies the persistent paper_trading.log into the RESULTADOS folder at every startup."""
        if not LOG_FILE.exists():
            return  # Nothing to copy on very first run
        try:
            from src.cloud.base_model.utils.config_utils import load_config
            from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_project
            _project_root = Path(__file__).parents[3]
            cfg = load_config()
            base_dir = get_drive_session_path("MODELOS", cfg)
            dest_dir = resolve_local_project(base_dir, _project_root).parent / "LOGS"
            dest_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_file = dest_dir / f"paper_trading_{ts}.log"
            shutil.copy2(LOG_FILE, dest_file)
            print(f"📋 Log backup saved to: {dest_file}")
        except Exception as e:
            print(f"⚠️  Could not backup log to RESULTADOS: {e}")

    _backup_log_to_resultados()
    asyncio.run(main())
