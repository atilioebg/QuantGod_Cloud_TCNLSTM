import polars as pl
import yaml
import logging
import os
import subprocess
import gc
import time
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is in path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import (
    get_pre_processed_dir, get_labelled_dir,
    get_logs_root
)

logger = logging.getLogger(__name__)

def process_single_file_labelling(pf: Path, config: dict, output_dir: Path):
    """
    Worker function to process a single file for labelling.
    """
    import psutil
    process = psutil.Process(os.getpid())
    
    try:
        labelling_cfg = config['pre_processing']['labelling']
        mins            = labelling_cfg.get('horizon_minutes', 15)
        use_first_touch = labelling_cfg.get('use_first_touch', True)
        pt_mult         = labelling_cfg.get('pt_multiplier')
        sl_mult         = labelling_cfg.get('sl_multiplier')
        vol_span        = labelling_cfg.get('vol_span', 100)
        target_bps_tp   = labelling_cfg.get('target_bps_tp')
        target_bps_sl   = labelling_cfg.get('target_bps_sl')
        
        # 1. Read
        df = pl.read_parquet(pf, memory_map=False)
        n_total = len(df)
        
        # 2. Logic
        df = df.with_columns([
            pl.col("close").log().diff().alias("__log_ret__"),
            pl.int_range(0, pl.len()).alias("__id__")
        ])
        
        try:
            df = df.with_columns(
                pl.col("__log_ret__").ewm_std(span=vol_span).over("island_id").alias("volatility_ewma")
            )
        except:
            df = df.with_columns([
                pl.col("__log_ret__").ewm_mean(span=vol_span).over("island_id").alias("_m1"),
                (pl.col("__log_ret__")**2).ewm_mean(span=vol_span).over("island_id").alias("_m2")
            ])
            df = df.with_columns(
                (pl.col("_m2") - pl.col("_m1")**2).clip(lower_bound=1e-12).sqrt().alias("volatility_ewma")
            ).drop(["_m1", "_m2"])

        avg_vol = df["volatility_ewma"].mean() or 0.0003
        if pt_mult is None and target_bps_tp is not None:
            pt_mult = (target_bps_tp / 10000.0) / avg_vol
        if sl_mult is None and target_bps_sl is not None:
            sl_mult = (target_bps_sl / 10000.0) / avg_vol

        pt_mult = pt_mult or 2.0
        sl_mult = sl_mult or 1.0

        df = df.with_columns(
            pl.from_epoch(pl.col("ts"), time_unit="ms").alias("_dt")
        )
        
        df_roll = df.rolling(
            index_column="_dt", 
            period=f"{mins}m", 
            offset="0s",
            group_by="island_id", 
            closed="both"
        ).agg([
            pl.col("__id__").last().alias("__id__"),
            pl.col("high").max().alias("fwd_max_high"),
            pl.col("low").min().alias("fwd_min_low"),
            pl.col("high").arg_max().alias("fwd_tp_idx"),
            pl.col("low").arg_min().alias("fwd_sl_idx"),
        ])
        
        df = df.join(df_roll, on=["__id__", "island_id"], how="left").drop(["_dt", "__id__"])
        
        df = df.with_columns([
            (pl.col("close") * (pl.col("volatility_ewma") * pt_mult).exp()).alias("barrier_up"),
            (pl.col("close") * (-pl.col("volatility_ewma") * sl_mult).exp()).alias("barrier_dn")
        ])
        
        stale_condition = pl.col("__stale_l2__") if "__stale_l2__" in df.columns else pl.lit(False)
        hits_tp = pl.col("fwd_max_high") >= pl.col("barrier_up")
        hits_sl = pl.col("fwd_min_low")  <= pl.col("barrier_dn")
        
        if use_first_touch:
            tp_first = pl.col("fwd_tp_idx")  <= pl.col("fwd_sl_idx")
            df = df.with_columns([
                pl.when(stale_condition).then(pl.lit(1, dtype=pl.Int8))
                .when(hits_tp & hits_sl & tp_first).then(pl.lit(2, dtype=pl.Int8))
                .when(hits_tp & hits_sl & ~tp_first).then(pl.lit(0, dtype=pl.Int8))
                .when(hits_tp).then(pl.lit(2, dtype=pl.Int8))
                .when(hits_sl).then(pl.lit(0, dtype=pl.Int8))
                .otherwise(pl.lit(1, dtype=pl.Int8)).alias("target")
            ])
        else:
            df = df.with_columns([
                pl.when(stale_condition).then(pl.lit(1, dtype=pl.Int8))
                .when(hits_tp & hits_sl).then(pl.lit(1, dtype=pl.Int8))
                .when(hits_tp).then(pl.lit(2, dtype=pl.Int8))
                .when(hits_sl).then(pl.lit(0, dtype=pl.Int8))
                .otherwise(pl.lit(1, dtype=pl.Int8)).alias("target")
            ])
            
        max_ts = df["ts"].max()
        horizon_ms = int(mins * 60 * 1000)
        df_final = df.filter(pl.col("ts") <= (max_ts - horizon_ms))
        
        out_path = output_dir / pf.name
        df_final.drop([c for c in [
            "__log_ret__", "volatility_ewma", "fwd_max_high", "fwd_min_low", 
            "barrier_up", "barrier_dn", "fwd_tp_idx", "fwd_sl_idx"
        ] if c in df_final.columns]).write_parquet(out_path)
        
        label_counts = {
            cls: len(df_final.filter(pl.col("target") == cls))
            for cls in [0, 1, 2]
        }
        
        mem_end = process.memory_info().rss / (1024 * 1024)
        return {"file": pf.name, "counts": label_counts, "n_kept": len(df_final), "n_total": n_total, "mem_mb": mem_end}
        
    except Exception as e:
        return {"file": pf.name, "error": str(e)}
    finally:
        if 'df' in locals(): del df
        if 'df_final' in locals(): del df_final
        gc.collect()

def run_labelling():
    # Setup logger & config
    base_config_path = Path("src/cloud/base_model/configs/master_config.yaml")
    if not base_config_path.exists():
        logger.error(f"Base Config file not found at {base_config_path}")
        return

    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    from src.cloud.base_model.utils.logging_utils import get_labelling_suffix
    suffix = get_labelling_suffix(config)
    setup_logger(config.get('naming_conventions', {}).get('labelling_log_prefix', "labelling"), suffix)

    input_dir  = Path(get_pre_processed_dir(config))
    output_dir = Path(get_labelled_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_dir.glob("*.parquet"))
    if not all_files:
        logger.error(f"No parquet files found in {input_dir}")
        return

    # Determinar Workers
    lab_cfg = config.get('pre_processing', {}).get('labelling', {})
    if lab_cfg.get('use_dynamic_workers', False):
        try: cpu_count = len(os.sched_getaffinity(0))
        except: cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count - 1)
    else:
        max_workers = lab_cfg.get('max_workers', 16)

    logger.info(f"🚀 Iniciando Labelling Paralelo: {len(all_files)} dias | {max_workers} workers | Memory-Safe Mode")

    global_counts = {0: 0, 1: 0, 2: 0}
    total_kept = 0
    total_raw = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file_labelling, pf, config, output_dir): pf for pf in all_files}
        
        for future in tqdm(as_completed(futures), total=len(all_files), desc="Labelling"):
            res = future.result()
            if "error" in res:
                logger.error(f"❌ Erro em {res.get('file', 'unknown')}: {res['error']}")
                continue
            
            for cls, count in res["counts"].items():
                global_counts[cls] += count
            total_kept += res["n_kept"]
            total_raw += res["n_total"]

    logger.info("Labelling phase finished.")
    logger.info("Final Label Distribution:")
    total_samples = sum(global_counts.values())
    for cls, name in sorted({0: "SELL", 1: "NEUTRAL", 2: "BUY"}.items()):
        ct = global_counts.get(cls, 0)
        pct = ct / total_samples * 100 if total_samples > 0 else 0
        logger.info(f"   {name} ({cls}): {ct:,} samples ({pct:.2f}%)")
    
    logger.info(f"Total labelled samples: {total_samples:,} (Raw: {total_raw:,} | Kept: {total_kept:,})")

    # ── 5. Export para o Drive → RESULTADOS_.../LABELLED/ ────────────────────
    try:
        from src.cloud.base_model.utils.path_utils import get_drive_session_path
        local_src   = str(output_dir)
        remote_dest = get_drive_session_path("LABELLED", config)
        rclone_cfg  = Path("rclone.conf")

        logger.info(f"🚀 Starting automated export to Drive: {remote_dest}...")

        # Pytest workers
        try:
            try: cpu_count = len(os.sched_getaffinity(0))
            except: cpu_count = os.cpu_count() or 1
            pytest_workers = min(max_workers, cpu_count)
        except Exception:
            pytest_workers = 1

        logger.info(f"🧪 Running Automated Health QA (pytest) with {pytest_workers} workers...")
        qa_log_path = output_dir / "labelling_health_QA.log"
        try:
            with open(qa_log_path, 'w', encoding='utf-8') as qa_file:
                subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/labelling/test_labelling_output.py", "-v", "-n", str(pytest_workers)],
                    stdout=qa_file,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ,
                             PRE_PROCESSED_DIR=str(input_dir),
                             LABELLED_DIR=local_src),
                    check=False
                )
            logger.info(f"✅ QA Report saved to {qa_log_path}")
        except Exception as e:
            logger.error(f"⚠️ QA Report generation failed: {e}")

        rclone_transfers = str(min(32, (os.cpu_count() or 4) * 2))
        cmd = ["rclone", "copy", local_src, remote_dest, "-P", "--transfers", rclone_transfers, "--checkers", rclone_transfers]
        if rclone_cfg.exists():
            cmd += ["--config", str(rclone_cfg)]
        if os.name == 'nt' and Path("rclone.exe").exists():
            cmd[0] = str(Path("rclone.exe").absolute())

        subprocess.run(cmd, check=True)
        logger.info(f"✅ Export completed successfully: {remote_dest}")
    except Exception as e:
        logger.error(f"❌ Automated export failed: {e}")

if __name__ == "__main__":
    run_labelling()
    # Audit Logs → DRIVE
    import yaml as _yaml
    with open("src/cloud/base_model/configs/master_config.yaml") as _f:
        _cfg = _yaml.safe_load(_f)
    upload_audit_to_drive(
        local_dirs=[f"{get_logs_root(_cfg)}/labelling"],
        stage_name="LABELLING",
        config=_cfg,
    )

