import subprocess
import time
import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - GUARDIAN - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('guardian')

def run_guardian():
    """
    Runs the paper trading script in an infinite loop.
    If the script crashes (e.g. OOM Killed or exception), it waits a few seconds and restarts it.
    If the user presses Ctrl+C, it exits the loop completely.
    """
    script_path = "src/cloud/execution/paper_trading_main.py"
    
    # Ensure the script is visible from the root
    if not Path(script_path).exists():
        logger.error(f"Cannot find script: {script_path}. Make sure you run guardian from the project root.")
        return

    logger.info("🛡️ QuantGod Guardian Watchdog Started.")
    logger.info(f"Target: {script_path}")
    logger.info("The system will automatically restart if an unexpected crash occurs.")
    logger.info("Press Ctrl+C to stop the Guardian and the trading script completely.")
    
    restart_count = 0
    
    while True:
        try:
            logger.info(f"🚀 Starting Paper Trading Process (Attempt #{restart_count + 1})...")
            
            # Start the subprocess and wait for it to finish
            # Using sys.executable ensures it uses the same python environment (venv)
            process = subprocess.Popen([sys.executable, script_path])
            process.wait()  # Blocks until the script exits randomly
            
            # Check why it exited
            if process.returncode == 0:
                logger.info("✅ Paper Trading exited gracefully (Return code 0). Stopping.")
                break
            else:
                logger.warning(f"⚠️ Process crashed or was killed! (Return code: {process.returncode})")
                logger.info("⏳ Waiting 10 seconds before restarting...")
                time.sleep(10)
                restart_count += 1
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Guardian received Ctrl+C. Initiating full shutdown...")
            if 'process' in locals() and process.poll() is None:
                logger.info("Terminating child process...")
                process.terminate()
                process.wait()
            logger.info("👋 Guardian offline.")
            break
        except Exception as e:
            logger.error(f"💥 Guardian caught an internal error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_guardian()
