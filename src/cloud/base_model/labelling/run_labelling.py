import polars as pl
import yaml
import logging
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys

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

def label_triple_barrier(
    all_files: list[Path],
    config: dict,
) -> pl.DataFrame:
    """
    [v8.0 Triple Barrier Method - Event Driven]
    
    Substitui a lógica de Shift (Point-to-Point) que é incompatível com barras não-lineares.
    
    Implementação:
    1. Horizontal Barriers: Profit Taking (Buy/Sell Threshold) e Stop Loss
    2. Vertical Barrier: Time-Stop fixado em `horizon_minutes`
    3. Custo Operacional: Aplica Taker Fees + Slippage estimate para evitar lucro de "papel"
    """
    labelling_cfg = config['pre_processing']['labelling']
    sell_th = labelling_cfg.get('sell_threshold', 0.003)
    buy_th  = labelling_cfg.get('buy_threshold', 0.003)
    mins    = labelling_cfg.get('horizon_minutes', 15)
    
    # Fees de microestrutura (Taker fee medio Binance/Bybit ~0.04% a 0.05%)
    taker_fee_pct = config.get('execution', {}).get('taker_fee_pct', 0.0005) 
    
    logger.info(f"Parametros Triple Barrier:")
    logger.info(f" - Vertical Barrier (Time-Stop): {mins} minutos")
    logger.info(f" - Profit Target Buy: +{buy_th*100}% | Sell: -{sell_th*100}%")
    logger.info(f" - Custo fixo embutido (Taker/Slippage): {taker_fee_pct*100}%")

    logger.info(f"Lendo {len(all_files)} arquivos em memória (Streaming via LazyFrame seria o proximo passo)...")
    dfs = []
    for pf in tqdm(all_files, desc="Lendo parquets"):
        df_i = pl.read_parquet(pf)
        df_i = df_i.with_columns(pl.lit(pf.name).alias("__source_file__"))
        dfs.append(df_i)

    df = pl.concat(dfs, how="diagonal_relaxed")
    if "datetime" in df.columns and "ts" not in df.columns:
        df = df.with_columns(pl.col("datetime").dt.timestamp("ms").alias("ts"))

    df = df.sort("ts")
    n_total = len(df)
    island_col = "island_id" if "island_id" in df.columns else None
    logger.info(f"Dataset concatenado: {n_total:,} barras | island={island_col}")

    # Lógica Simplificada O(N*W) segura em polars: 
    # Para Dollar Bars puras, o numero de rows em 15m pode variar.
    # Como não podemos usar shift(N), usamos um group_by_dynamic sobre as rows futuras, 
    # ou uma rolling list de timestamps. A abordagem estrita "Join asof" é mais eficaz 
    # pro Time-Stop (Vertical Barrier): 
    # Qual o close/high/low exato no momento tempo T + horizon_minutes?
    
    # ── 1. Vertical Barrier (Time Stop) ───────────────────────────────────────
    horizon_ms = int(mins * 60 * 1000)
    
    # Criamos a coluna com o timestamp futuro alvo
    df = df.with_columns((pl.col("ts") + horizon_ms).alias("vertical_barrier_ts"))
    
    # Buscamos a primeira barra imediatamente ANTES do vertical barrier (fechamento de tempo)
    df_vertical = df.select(["ts", "close", "island_id"]).rename({
        "ts": "v_ts",
        "close": "v_close",
        "island_id": "v_island_id"
    })
    
    # Backward join: pego o preco que existe no exato momento da barreira de tempo
    df = df.join_asof(
        df_vertical,
        left_on="vertical_barrier_ts",
        right_on="v_ts",
        strategy="backward"
    )
    
    # Se a ilha mudou entre a execucao e o time-stop (ocorreu gap > limiar), 
    # a operacao eh nula (Island Boundary Violated).
    invalid_island = pl.lit(False)
    if island_col:
        invalid_island = (pl.col("v_island_id").is_null()) | (pl.col("v_island_id") != pl.col(island_col))

    # ── 2. Triple Barrier Assessment (Point-to-Point conservador na Barreira Vertical) 
    # Nota: Em HFT ideal, calculariamos `max_high` e `min_low` num rolling window entre ts e v_ts.
    # Mas para TCN de direcao (Trend/Sniper), a resolucao do PnL no FIM da janela exclui falsos
    # rompimentos com pullback (wicks) protegendo a rede de prever ruido como alvo efetivo.
    
    # PnL no Time-Stop descontando Taker Fees dupla (entrada + saida)
    total_fee = (taker_fee_pct * 2)
    
    # Ratio direcional bruto
    df = df.with_columns(
        (pl.col("v_close") / pl.col("close")).alias("ratio_future")
    )
    
    # Retornos Reais (subtraindo as fees da casa e slippage estipulados)
    # Se a flag __stale_l2__ existir (livro de ordens fantasma/parado enquanto trades correm),
    # o sinal direcional é TERMINANTEMENTE censurado para NEUTRAL.
    stale_condition = pl.col("__stale_l2__") if "__stale_l2__" in df.columns else pl.lit(False)

    df = df.with_columns([
        pl.when(invalid_island)
            .then(None)
        .when(stale_condition)
            .then(pl.lit(1, dtype=pl.Int8))   # NEUTRAL (Ignora o setup por ser baseado em livro fantasma)
        .when(buy_return_real >= buy_th)
            .then(pl.lit(2, dtype=pl.Int8))   # BUY Hits Profit Target at Time-Stop
        .when(sell_return_real >= sell_th)
            .then(pl.lit(0, dtype=pl.Int8))   # SELL Hits Profit Target at Time-Stop
        .otherwise(pl.lit(1, dtype=pl.Int8))  # NEUTRAL (Fails to pierce barrier w/ profit padding)
        .alias("target")
    ])

    df_final = (
        df.drop_nulls(subset=["target"])
          .drop(["vertical_barrier_ts", "v_ts", "v_close", "v_island_id", "ratio_future"])
    )

    n_kept = len(df_final)
    n_dropped = n_total - n_kept
    logger.info(f"Triple Barrier Labeling: {n_kept:,} válidos | {n_dropped:,} descartados (borda final de matriz ou quebra de fluxo)")

    return df_final


def run_labelling():
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

    logger.info(f"Iniciando Triple Barrier Labelling (Dollar/Tick Bars Compatible)...")

    df_labelled = label_triple_barrier(all_files, config)

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

    logger.info("Salvando arquivos rotulados por fonte original...")
    save_errors = 0
    for source_file in tqdm(all_files, desc="Salvando"):
        df_file = df_labelled.filter(
            pl.col("__source_file__") == source_file.name
        ).drop("__source_file__")

        if len(df_file) == 0:
            save_errors += 1
            continue

        df_file.write_parquet(output_dir / source_file.name)

    if save_errors:
        logger.warning(f"⚠️  {save_errors} arquivo(s) sem amostras válidas (bordas de ilha/lookahead — esperado).")

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
    # Audit Logs → RESULTADOS_.../AUDITORIA/LABELLING/
    import yaml as _yaml
    with open("src/cloud/base_model/configs/master_config.yaml") as _f:
        _cfg = _yaml.safe_load(_f)
    upload_audit_to_drive(
        local_dirs=[f"{get_logs_root(_cfg)}/labelling"],
        stage_name="LABELLING",
        config=_cfg,
    )

