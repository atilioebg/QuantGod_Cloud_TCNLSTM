import polars as pl
import yaml
import logging
import os
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure project root is in path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger, get_labelling_suffix, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import (
    get_pre_processed_dir, get_labelled_dir,
    get_logs_root
)

logger = logging.getLogger(__name__)


def label_full_dataset(
    all_files: list[Path],
    config: dict,
) -> pl.DataFrame:
    """
    [v5.0 Gold — Continuous Dataset Labelling]

    Concatena TODOS os arquivos num único DataFrame antes de calcular os labels.
    Isso garante que shift(-h).over(island_id) possa atravessar fronteiras de
    arquivo, recuperando ~3 amostras por dia que antes eram descartadas por
    processar cada arquivo isoladamente.

    Regras de Target (Point-to-Point Pure):
    - BUY  (2): close[t+h] / close[t] >= 1 + buy_th    (dentro da mesma ilha)
    - SELL (0): close[t+h] / close[t] <= 1 - sell_th   (dentro da mesma ilha)
    - NEUTRAL (1): caso contrário
    - NaN / drop: future_close é null (fim de dataset ou cruzamento de ilha)

    Nota: tick_count=0 é BTC quieto — dado válido. Target legítimo (NEUTRAL).
    A proteção por island_id já cobre todos os gaps reais de dados.
    """
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    resample_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
    resample_min  = int(pd.to_timedelta(resample_freq).total_seconds() // 60)
    lookahead_bars = max(1, mins // resample_min)

    logger.info(f"Lookahead: {mins}min / {resample_freq} = {lookahead_bars} barras")

    # ── 1. Ler e concatenar todos os arquivos ────────────────────────────────
    logger.info(f"Lendo {len(all_files)} arquivos em memória...")
    dfs = []
    for pf in tqdm(all_files, desc="Lendo parquets"):
        df_i = pl.read_parquet(pf)
        df_i = df_i.with_columns(pl.lit(pf.name).alias("__source_file__"))
        dfs.append(df_i)

    df = pl.concat(dfs, how="diagonal")

    # Garantir ordem cronológica (necessário para shift funcionar corretamente)
    if "timestamp" in df.columns:
        df = df.sort("timestamp")
    elif "datetime" in df.columns:
        df = df.sort("datetime")

    n_total = len(df)
    island_col = "island_id" if "island_id" in df.columns else None
    logger.info(f"Dataset concatenado: {n_total:,} barras | island_col={island_col}")

    # ── 2. Colunas futuras via shift no dataset completo ─────────────────────
    if island_col:
        df = df.with_columns([
            # Preço futuro exato (t + h) dentro da mesma ilha
            pl.col("close").shift(-lookahead_bars).over(island_col).alias("future_close"),
            # Island_id futura: se mudou → fronteira de ilha → invalida
            pl.col(island_col).shift(-lookahead_bars).over(island_col).alias("future_island_id"),
        ])
    else:
        df = df.with_columns([
            pl.col("close").shift(-lookahead_bars).alias("future_close"),
        ])
        df = df.with_columns(pl.lit(None).alias("future_island_id"))

    # ── 3. Cálculo do Target (Point-to-Point Pure) ───────────────────────────
    island_boundary_violated = (
        pl.col("future_island_id").is_null() |
        (pl.col("future_island_id") != pl.col(island_col))
    ) if island_col else pl.lit(False)

    df = df.with_columns([
        pl.when(pl.col("future_close").is_null())
            .then(None)
        .when(island_boundary_violated)
            .then(None)
        .when(pl.col("future_close") >= pl.col("close") * (1.0 + buy_th))
            .then(pl.lit(2, dtype=pl.Int8))   # BUY
        .when(pl.col("future_close") <= pl.col("close") * (1.0 - sell_th))
            .then(pl.lit(0, dtype=pl.Int8))   # SELL
        .otherwise(pl.lit(1, dtype=pl.Int8))  # NEUTRAL
        .alias("target")
    ])

    # ── 4. Remover apenas as últimas h linhas do dataset inteiro ─────────────
    # (não mais 3 linhas × 1.133 arquivos = 3.399 — agora são só 3 linhas no total)
    df_final = (
        df.slice(0, n_total - lookahead_bars)
          .drop_nulls(subset=["target"])
          .drop([c for c in ["future_close", "future_island_id"] if c in df.columns])
    )

    n_kept    = len(df_final)
    n_dropped = n_total - n_kept
    logger.info(f"Labels calculados: {n_kept:,} válidos | {n_dropped:,} descartados (bordas de ilha + lookahead)")

    return df_final


def run_labelling():
    # ── 1. Load Config ────────────────────────────────────────────────────────
    base_config_path = Path("src/cloud/base_model/configs/master_config.yaml")
    if not base_config_path.exists():
        logger.error(f"Base Config file not found at {base_config_path}")
        return

    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    suffix  = f"_labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    setup_logger("labelling", suffix)

    input_dir  = Path(get_pre_processed_dir(config))
    output_dir = Path(get_labelled_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_dir.glob("*.parquet"))
    if not all_files:
        logger.error(f"No parquet files found in {input_dir}")
        return

    logger.info(f"Found {len(all_files)} files to label.")
    logger.info("Mode: Continuous dataset labelling (cross-file island continuity preserved).")

    # ── 2. Labelling Contínuo ─────────────────────────────────────────────────
    df_labelled = label_full_dataset(all_files, config)

    # ── 3. Distribuição final ─────────────────────────────────────────────────
    label_counts = {
        row["target"]: row["count"]
        for row in df_labelled["target"].value_counts().to_dicts()
    }
    total_samples = sum(label_counts.values())

    logger.info("Labelling phase finished.")
    logger.info("Final Label Distribution:")
    for cls, name in sorted({0: "SELL", 1: "NEUTRAL", 2: "BUY"}.items()):
        ct = label_counts.get(cls, 0)
        pct = ct / total_samples * 100 if total_samples > 0 else 0
        logger.info(f"   {name} ({cls}): {ct:,} samples ({pct:.2f}%)")
    logger.info(f"Total labelled samples: {total_samples:,}")
    logger.info(f"Total processed files: {len(all_files)}")

    # ── 4. Salvar — split de volta por arquivo de origem ─────────────────────
    logger.info("Salvando arquivos rotulados por fonte original...")
    save_errors = 0
    for source_file in tqdm(all_files, desc="Salvando"):
        df_file = df_labelled.filter(
            pl.col("__source_file__") == source_file.name
        ).drop("__source_file__")

        if len(df_file) == 0:
            logger.warning(f"⚠️  {source_file.name}: 0 amostras após labelling (arquivo pode ser borda de ilha)")
            save_errors += 1
            continue

        df_file.write_parquet(output_dir / source_file.name)

    if save_errors:
        logger.warning(f"⚠️  {save_errors} arquivo(s) sem amostras válidas (bordas de ilha — esperado).")

    # ── 5. Export para o Drive → RESULTADOS_.../LABELLED/ ────────────────────
    try:
        from src.cloud.base_model.utils.path_utils import get_drive_session_path
        local_src   = str(output_dir)
        remote_dest = get_drive_session_path("LABELLED", config)
        rclone_cfg  = Path("rclone.conf")

        logger.info(f"🚀 Starting automated export to Drive: {remote_dest}...")

        # QA Tests
        logger.info("🧪 Running Automated Health QA (pytest)...")
        qa_log_path = output_dir / "labelling_health_QA.log"
        try:
            with open(qa_log_path, 'w', encoding='utf-8') as qa_file:
                subprocess.run(
                    ["pytest", "tests/labelling/test_labelling_output.py", "-v"],
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

        cmd = ["rclone", "copy", local_src, remote_dest, "-P"]
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
    # Audit Logs → RESULTADOS_.../AUDITORIA/LABELLING/
    import yaml as _yaml
    with open("src/cloud/base_model/configs/master_config.yaml") as _f:
        _cfg = _yaml.safe_load(_f)
    upload_audit_to_drive(
        local_dirs=[f"{get_logs_root(_cfg)}/labelling"],
        stage_name="LABELLING",
        config=_cfg,
    )
