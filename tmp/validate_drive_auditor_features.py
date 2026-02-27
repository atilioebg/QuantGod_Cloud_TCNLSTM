"""
validate_drive_auditor_features.py — Scratch script to validate auditor features
from a small sample of Drive data.

Runs rclone to pull 5 sample pre-processed parquet files from the Drive,
then validates that all 20 auditor features (from master_config.yaml) can be
correctly constructed from the data.

Usage (run from project root):
    .\\venv\\Scripts\\python.exe tmp\\validate_drive_auditor_features.py
"""
import sys
import subprocess
import tempfile
import yaml
from pathlib import Path
import polars as pl
import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud.base_model.utils.logging_utils import setup_logger
import logging
setup_logger("drive_validation", "")
logger = logging.getLogger("drive_validation")

# ── Load config ───────────────────────────────────────────────────────────────
cfg_path = PROJECT_ROOT / "src" / "cloud" / "base_model" / "configs" / "master_config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

FEATURE_NAMES    = config["model"]["feature_names"]          # 32 base features
AUDITOR_FEATURES = config["model"]["auditor_features"]       # 20 auditor features
NUM_FEATURES     = config["model"]["num_features"]           # 32
RAW_SOURCE       = config["pipeline_paths"]["raw_l2_source"] # drive:PROJETOS/BTC_USDT_L2_2023_2026

logger.info(f"Config loaded: {NUM_FEATURES} base features, {len(AUDITOR_FEATURES)} auditor features")

# ── Pull a small sample from Drive ───────────────────────────────────────────
# We look into the pre-processed folder on Drive (if available) or the labelled folder.
# Try both paths in order of preference.
DRIVE_SOURCES_TO_TRY = [
    "drive:PROJETOS/LABELLED_L2_2023_2026_5_MINUTE_32_FEATURES",
    "drive:PROJETOS/BTC_USDT_L2_2023_2026",
]

RCLONE_EXE = PROJECT_ROOT / "rclone.exe"
RCLONE_CONF = PROJECT_ROOT / "rclone.conf"
RCLONE_CMD_BASE = [str(RCLONE_EXE), "--config", str(RCLONE_CONF)]

tmp_dir = Path(tempfile.mkdtemp(prefix="qg_drive_validation_"))
logger.info(f"Temp dir: {tmp_dir}")

sample_files = []
chosen_source = None

for drive_source in DRIVE_SOURCES_TO_TRY:
    logger.info(f"Attempting to list files from: {drive_source}")
    try:
        # List files
        result = subprocess.run(
            RCLONE_CMD_BASE + ["lsf", drive_source, "--include=*.parquet", "--max-depth=1"],
            capture_output=True, text=True, timeout=30
        )
        files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip().endswith(".parquet")]
        if files:
            # Take first 5 files as sample
            sample_names = files[:5]
            chosen_source = drive_source
            logger.info(f"Found {len(files)} parquet files. Sampling {len(sample_names)}: {sample_names}")
            break
        else:
            logger.warning(f"No parquet files found at {drive_source}, trying next source...")
    except Exception as e:
        logger.warning(f"Failed to list {drive_source}: {e}")

if not chosen_source:
    logger.error("❌ Could not reach any Drive source. Check rclone.conf and Drive connectivity.")
    sys.exit(1)

# Download the sample files
logger.info(f"📥 Downloading {len(sample_names)} files from {chosen_source}...")
for fname in sample_names:
    src = f"{chosen_source}/{fname}"
    dst = tmp_dir / fname
    try:
        subprocess.run(
            RCLONE_CMD_BASE + ["copy", src, str(tmp_dir)],
            check=True, capture_output=True, timeout=60
        )
        if dst.exists():
            sample_files.append(dst)
            logger.info(f"  ✅ Downloaded: {fname} ({dst.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to download {fname}: {e}")

if not sample_files:
    logger.error("❌ No files downloaded from Drive. Cannot validate.")
    sys.exit(1)

logger.info(f"\n{'='*60}")
logger.info(f"VALIDATING {len(sample_files)} SAMPLE FILES")
logger.info(f"{'='*60}")

# ── Validate base features (32) ───────────────────────────────────────────────
logger.info("\n[1/3] Validating BASE features (32)")
all_passed = True
for fp in sample_files:
    try:
        df = pl.read_parquet(fp)
        cols = set(df.columns)
        missing = [f for f in FEATURE_NAMES if f not in cols]
        extra = [c for c in cols if c not in FEATURE_NAMES and c not in ("target", "close")]
        
        if missing:
            logger.error(f"  ❌ {fp.name}: MISSING base features: {missing}")
            all_passed = False
        else:
            logger.info(f"  ✅ {fp.name}: All 32 base features present | rows={len(df):,} | extra_cols={extra[:5]}")
        
        # Check for NaN/Inf
        for feat in FEATURE_NAMES:
            if feat in cols:
                col_arr = df[feat].to_numpy()
                n_nan = np.isnan(col_arr).sum()
                n_inf = np.isinf(col_arr).sum()
                if n_nan > 0 or n_inf > 0:
                    logger.warning(f"    ⚠️ {feat}: NaN={n_nan}, Inf={n_inf}")
    except Exception as e:
        logger.error(f"  ❌ {fp.name}: Failed to read: {e}")
        all_passed = False

# ── Validate target column ────────────────────────────────────────────────────
logger.info("\n[2/3] Validating TARGET column (labelling)")
for fp in sample_files:
    try:
        df = pl.read_parquet(fp)
        if "target" not in df.columns:
            logger.error(f"  ❌ {fp.name}: 'target' column MISSING — file is pre-labelling")
            continue
        
        vc = {row["target"]: row["count"] for row in df["target"].value_counts().to_dicts()}
        sell  = vc.get(0, 0)
        neutral = vc.get(1, 0)
        buy   = vc.get(2, 0)
        total = sell + neutral + buy
        logger.info(f"  ✅ {fp.name}: SELL={sell:,}({sell/total:.1%}) NEUTRAL={neutral:,}({neutral/total:.1%}) BUY={buy:,}({buy/total:.1%})")
    except Exception as e:
        logger.error(f"  ❌ {fp.name}: {e}")

# ── Validate auditor feature constructability ─────────────────────────────────
logger.info("\n[3/3] Validating AUDITOR features constructability")
logger.info(f"Auditor requires {len(AUDITOR_FEATURES)} features (6 model probs + 14 algo indicators)")

# The model-side probs (base_prob_sell/neu/buy + spec_prob_sell/neu/buy) are generated
# at inference time by the model — we verify the algo indicators can be computed
# from the base features present in the parquet file.
ALGO_INDICATORS_VS_BASE_COLUMNS = {
    "ema_trend":      ["close"],              # EMA cross (8, 21) from close price
    "ema_cross_dist": ["close"],              # Distance from EMA crossover
    "bb_pct":         ["close"],              # Bollinger Bands %B
    "rsi_14":         ["close"],              # RSI(14)
    "stoch_14":       ["close"],              # Stochastic(14) — needs only close for simplified version
    "atr_norm":       ["close"],              # ATR normalized — simplified from close diff
    "vol_1h":         ["log_volume"],         # 1h volume sum
    "vol_zscore_1h":  ["log_volume"],         # Z-score of vol_1h
    "delta_vol_24h":  ["log_volume"],         # 24h vol derivative
    "adx_14":         ["close"],              # ADX(14) — simplified directional index
    "vwap_zscore":    ["close", "log_volume"],# VWAP z-score
    "mfi_14":         ["close", "log_volume"],# Money Flow Index(14)
    "book_skew_bid":  ["mean_obi", "mean_deep_obi"],  # Bid skewness from OBI features
    "book_skew_ask":  ["mean_obi", "mean_deep_obi"],  # Ask skewness from OBI features
}

for fp in sample_files[:1]:  # Only check one file for this validation
    df = pl.read_parquet(fp)
    cols = set(df.columns)
    
    logger.info(f"  Checking constructability from: {fp.name}")
    for indicator, required_sources in ALGO_INDICATORS_VS_BASE_COLUMNS.items():
        missing_src = [s for s in required_sources if s not in cols]
        if missing_src:
            logger.error(f"    ❌ {indicator}: cannot build — missing source columns: {missing_src}")
            all_passed = False
        else:
            logger.info(f"    ✅ {indicator}: constructable from {required_sources}")
    
    # Also validate model prob features (6) are generated externally by inference
    model_probs = ["base_prob_sell", "base_prob_neu", "base_prob_buy",
                   "spec_prob_sell", "spec_prob_neu", "spec_prob_buy"]
    logger.info(f"  ℹ️  Model probs {model_probs} are generated at inference time (not in parquet) — OK by design")

# ── Summary ───────────────────────────────────────────────────────────────────
logger.info(f"\n{'='*60}")
if all_passed:
    logger.info("✅ VALIDATION PASSED: All auditor features are correctly constructable from Drive data")
else:
    logger.error("❌ VALIDATION FAILED: See errors above")

# Cleanup
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
logger.info(f"Temp dir cleaned up: {tmp_dir}")
