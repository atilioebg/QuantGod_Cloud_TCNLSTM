import polars as pl
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import os
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is in path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger, get_labelling_suffix, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import (
    get_pre_processed_dir, get_labelled_dir,
    get_drive_dir, get_logs_root
)

logger = logging.getLogger(__name__)

def apply_labelling(file_path, config, audit_mode: bool = False):
    """
    [v4.9 Gold — Point-to-Point Pure]
    Labelling estritamente close-to-close no horizonte exato de lookahead.

    Regras:
    - Target BUY  (2): close[t+h] / close[t] >= 1 + buy_th
    - Target SELL (0): close[t+h] / close[t] <= 1 - sell_th
    - Target NEUTRAL(1): caso contrário
    - Target NaN: qualquer gap (tick_count=0) no intervalo [t+1, t+h],
                  OU se t+h pertencer a uma island_id diferente de t.

    HIGH e LOW não são usados em nenhuma etapa.
    """
    try:
        sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
        buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
        mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)

        # Converter horizonte para BARRAS (ex: 15min / 5min = 3 barras)
        resample_freq  = config['pre_processing']['etl'].get('resample_freq', '1min')
        resample_min   = int(pd.to_timedelta(resample_freq).total_seconds() // 60)
        lookahead_bars = max(1, mins // resample_min)

        # Diretório de saída (genérico, definido no master_config)
        output_dir = Path(get_labelled_dir(config))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Carregar Parquet
        df = pl.read_parquet(file_path)

        island_col = 'island_id' if 'island_id' in df.columns else None

        # 2. Colunas futuras — SOMENTE close e tick_count; HIGH e LOW ignorados.
        if island_col:
            df = df.with_columns([
                # Preço futuro exato (t + lookahead_bars) dentro da mesma ilha
                pl.col("close").shift(-lookahead_bars).over(island_col).alias("future_close"),
                # Island_id futura: se mudou → fronteira violada → NaN no target
                pl.col(island_col).shift(-lookahead_bars).over(island_col).alias("future_island_id"),
                # Zero Tolerance: qualquer gap no caminho invalida o sample
                pl.col("tick_count").rolling_min(window_size=lookahead_bars)
                    .shift(-lookahead_bars).over(island_col).alias("future_min_ticks"),
            ])
        else:
            df = df.with_columns([
                pl.col("close").shift(-lookahead_bars).alias("future_close"),
                pl.col("tick_count").rolling_min(window_size=lookahead_bars)
                    .shift(-lookahead_bars).alias("future_min_ticks"),
            ])
            # Sem island_col, não há proteção de fronteira — adiciona coluna nula
            df = df.with_columns(pl.lit(None).alias("future_island_id"))

        # 3. Target — Point-to-Point Pure
        island_boundary_violated = (
            pl.col("future_island_id").is_null() |
            (pl.col("future_island_id") != pl.col(island_col))
        ) if island_col else pl.lit(False)

        df = df.with_columns([
            pl.when(pl.col("future_min_ticks").is_null())
                .then(None)  # Fim de ilha (shift retornou null)
            .when(pl.col("future_min_ticks") == 0)
                .then(None)  # ZERO TOLERANCE: gap no caminho
            .when(island_boundary_violated)
                .then(None)  # ISLAND BOUNDARY: t+h está em ilha diferente
            .when(pl.col("future_close") >= pl.col("close") * (1.0 + buy_th))
                .then(pl.lit(2, dtype=pl.Int8))  # BUY
            .when(pl.col("future_close") <= pl.col("close") * (1.0 - sell_th))
                .then(pl.lit(0, dtype=pl.Int8))  # SELL
            .otherwise(pl.lit(1, dtype=pl.Int8))  # NEUTRAL
            .alias("target")
        ])

        # 4. Taxa de invalidação (antes do drop)
        total_valid_rows  = len(df) - lookahead_bars
        invalidated_rows  = df.slice(0, total_valid_rows).filter(pl.col("target").is_null()).height
        invalidation_rate = invalidated_rows / total_valid_rows if total_valid_rows > 0 else 0

        # 5. Cleanup: remove lookahead tail + amostras inválidas + colunas auxiliares
        drop_cols = ["future_close", "future_island_id", "future_min_ticks"]
        df_final = (
            df.slice(0, total_valid_rows)
              .drop_nulls(subset=["target"])
              .drop([c for c in drop_cols if c in df.columns])
        )

        # 6. Salvar
        output_path = output_dir / file_path.name
        df_final.write_parquet(output_path)

        counts = {row['target']: row['count'] for row in df_final['target'].value_counts().to_dicts()}

        if audit_mode:
            return {
                "status": "success", "file": file_path.name,
                "invalidation_rate": invalidation_rate,
                "counts": counts,
                "df_with_target": df.slice(0, total_valid_rows),  # para comparação externa
            }

        return {
            "status": "success",
            "file": file_path.name,
            "invalidation_rate": invalidation_rate,
            "counts": counts,
        }

    except Exception as e:
        return {
            "status": "error",
            "file": file_path.name,
            "error": str(e),
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
    
    # Output path based on pipeline root + unified prefix
    base_output = Path(get_labelled_dir(config))
    
    # Diretório de entrada: pasta dinâmica PRE_PROCESSED_L2_{horizon}_{lookback}_{freq}
    # Mesmo padrão usado pelo run_pipeline.py no DataLoader e no export para o Drive.
    input_dir = Path(get_pre_processed_dir(config))
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

    # 4. Automated Export to Google Drive
    try:
        local_src   = str(base_output)
        remote_dest = get_drive_dir(
            config['pipeline_paths'].get('drive_labelled_remote', 'drive:PROJETOS/LABELLED_L2'),
            config
        )
        rclone_cfg = Path("rclone.conf")

        logger.info(f"🚀 Starting automated export to Drive: {remote_dest}...")

        # --- Run QA Tests Before Export ---
        logger.info("🧪 Running Automated Health QA (pytest)...")
        qa_log_path = Path(local_src) / "labelling_health_QA.log"
        qa_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(qa_log_path, 'w', encoding='utf-8') as qa_file:
                subprocess.run(
                    ["pytest", "tests/labelling/test_labelling_output.py", "-v"],
                    stdout=qa_file,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PRE_PROCESSED_DIR=str(input_dir), LABELLED_DIR=local_src),
                    check=False
                )
            logger.info(f"✅ QA Report saved to {qa_log_path}")
        except Exception as e:
            logger.error(f"⚠️ QA Report generation failed: {e}")

        cmd = ["rclone", "copy", str(local_src), remote_dest, "-P"]
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
    # Audit Logs → Drive  (PROJETOS/AUDITORIA_.../LABELLING)
    import yaml as _yaml
    with open("src/cloud/base_model/configs/master_config.yaml") as _f:
        _cfg = _yaml.safe_load(_f)
    upload_audit_to_drive(
        local_dirs=[f"{get_logs_root(_cfg)}/labelling"],
        stage_name="LABELLING",
    )
