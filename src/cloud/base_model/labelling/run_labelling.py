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
    [v8.2 Triple Barrier Method - High Fidelity Event Driven]
    
    Implementação estrutural de Marcos López de Prado:
    1. Volatilidade EWMA Dinâmica como base de escala.
    2. Barreiras Horizontais (Take Profit / Stop Loss) via rolling peak/trough.
    3. Barreira Vertical (Time-Stop) corrigida via ts real do config.
    4. Causalidade estrita e tratamento de 'crossed/stale' books.
    """
    labelling_cfg = config['pre_processing']['labelling']
    
    mins            = labelling_cfg.get('horizon_minutes', 15)
    use_first_touch = labelling_cfg.get('use_first_touch', True)
    pt_mult         = labelling_cfg.get('pt_multiplier')
    sl_mult         = labelling_cfg.get('sl_multiplier')
    vol_span        = labelling_cfg.get('vol_span', 100)
    
    # Alvos Nominais em Basis Points (bps)
    target_bps_tp   = labelling_cfg.get('target_bps_tp')
    target_bps_sl   = labelling_cfg.get('target_bps_sl')
    
    # Fees e Custos
    taker_fee_pct = config.get('execution', {}).get('taker_fee_pct', 0.0005) 
    
    logger.info("Configurando Motor de Labelling Prado-IID:")
    logger.info(f" - Janela Vertical: {mins} min (Cronológica)")
    logger.info(f" - Multiplicadores Vol: PT={pt_mult}x | SL={sl_mult}x")
    logger.info(f" - Span Volatilidade (EWMA): {vol_span} bars")
    
    logger.info(f"Processando {len(all_files)} fragmentos...")
    dfs = []
    for pf in tqdm(all_files, desc="Lendo parquets"):
        df_i = pl.read_parquet(pf)
        df_i = df_i.with_columns(pl.lit(pf.name).alias("__source_file__"))
        dfs.append(df_i)

    df = pl.concat(dfs, how="diagonal_relaxed")
    df = df.sort("ts")
    
    # ── 1. Cálculo da Volatilidade Diária EWMA ──────────────────────────────
    # Usamos o log retorno do close para normalizar a escala
    df = df.with_columns([
        pl.col("close").log().diff().alias("__log_ret__"),
        pl.arange(0, pl.len()).alias("__id__") # Unique ID to prevent join explosion
    ])
    
    # Volatilidade adaptativa (EWMA Std) - calculada com fallback caso versao polars seja antiga
    try:
        df = df.with_columns(
            pl.col("__log_ret__").ewm_std(span=vol_span).over("island_id").alias("volatility_ewma")
        )
    except:
        # Fallback manual: Var(X) = E[X^2] - (E[X])^2
        df = df.with_columns([
            pl.col("__log_ret__").ewm_mean(span=vol_span).over("island_id").alias("_m1"),
            (pl.col("__log_ret__")**2).ewm_mean(span=vol_span).over("island_id").alias("_m2")
        ])
        df = df.with_columns(
            (pl.col("_m2") - pl.col("_m1")**2).clip(lower_bound=1e-12).sqrt().alias("volatility_ewma")
        ).drop(["_m1", "_m2"])

    # ── 1b. Cálculo Dinâmico de Multiplicadores via BPS (se aplicável) ───────
    # Se pt_multiplier não for definido, buscamos atingir target_bps_tp na média.
    avg_vol = df["volatility_ewma"].mean() or 0.0003 # Fallback 3 bps
    
    if pt_mult is None and target_bps_tp is not None:
        pt_mult = (target_bps_tp / 10000.0) / avg_vol
        logger.info(f"🎯 Dynamic PT Multiplier: {pt_mult:.4f} (Target: {target_bps_tp} bps | Avg Vol: {avg_vol*10000:.2f} bps)")
    
    if sl_mult is None and target_bps_sl is not None:
        sl_mult = (target_bps_sl / 10000.0) / avg_vol
        logger.info(f"🎯 Dynamic SL Multiplier: {sl_mult:.4f} (Target: {target_bps_sl} bps | Avg Vol: {avg_vol*10000:.2f} bps)")

    # Fallback caso nada seja definido
    pt_mult = pt_mult or 2.0
    sl_mult = sl_mult or 1.0

    # ── 2. Scanning Futuro (Rolling Windows Forward) ─────────────────────────
    # Em Polars, para olhar pra frente (lookahead), usamos offset=0 e period=mins
    # Para garantir o uso de strings de duração ("15m"), convertemos ts para Datetime
    df = df.with_columns(
        pl.from_epoch(pl.col("ts"), time_unit="ms").alias("_dt")
    )
    
    # Executamos a rolagem frontal para capturar picos e vales do "futuro"
    # A janela é [T, T + horizon]
    df_roll = df.rolling(
        index_column="_dt", 
        period=f"{mins}m", 
        offset="0s",
        group_by="island_id", 
        closed="both"
    ).agg([
        pl.col("__id__").last().alias("__id__"), # Keep the ID of the current row
        pl.col("high").max().alias("fwd_max_high"),
        pl.col("low").min().alias("fwd_min_low"),
        pl.col("close").last().alias("fwd_close_at_stop"),
        pl.col("high").arg_max().alias("fwd_tp_idx"),  # Índice relativo do toque do TP na janela
        pl.col("low").arg_min().alias("fwd_sl_idx"),   # Índice relativo do toque do SL na janela
    ])
    
    # Re-acoplamos os resultados via join no timestamp original
    df = df.join(df_roll, on=["__id__", "island_id"], how="left").drop(["_dt", "__id__"])
    df = df.sort("ts")
    
    # ── 3. Definição das Barreiras Reais ─────────────────────────────────────
    # Usamos o modelo log-normal: Preço_Futuro = Preço_T * exp(vol * multiplicador)
    df = df.with_columns([
        (pl.col("close") * (pl.col("volatility_ewma") * pt_mult).exp()).alias("barrier_up"),
        (pl.col("close") * (-pl.col("volatility_ewma") * sl_mult).exp()).alias("barrier_dn")
    ])
    
    # ── 4. Lógica de Decisão (Triple Barrier) ─────────────────────────────────
    stale_condition = pl.col("__stale_l2__") if "__stale_l2__" in df.columns else pl.lit(False)
    hits_tp = pl.col("fwd_max_high") >= pl.col("barrier_up")
    hits_sl = pl.col("fwd_min_low")  <= pl.col("barrier_dn")
    
    if use_first_touch:
        # [v9.0 AFML-Aligned] Quando ambas as barreiras são atingidas na janela,
        # a barreira tocada PRIMEIRO (menor índice relativo) determina o label.
        tp_first = pl.col("fwd_tp_idx")  <= pl.col("fwd_sl_idx")
        
        df = df.with_columns([
            pl.when(stale_condition)
                .then(pl.lit(1, dtype=pl.Int8))
            .when(hits_tp & hits_sl & tp_first)
                .then(pl.lit(2, dtype=pl.Int8))           # TP tocou primeiro → BUY
            .when(hits_tp & hits_sl & ~tp_first)
                .then(pl.lit(0, dtype=pl.Int8))           # SL tocou primeiro → SELL
            .when(hits_tp)
                .then(pl.lit(2, dtype=pl.Int8))           # BUY
            .when(hits_sl)
                .then(pl.lit(0, dtype=pl.Int8))           # SELL
            .otherwise(pl.lit(1, dtype=pl.Int8))          # Time-Stop → NEUTRAL
            .alias("target")
        ])
    else:
        # [v1.0 Legado] Ambos atingidos na mesma janela? → Neutro (Conflito)
        df = df.with_columns([
            pl.when(stale_condition)
                .then(pl.lit(1, dtype=pl.Int8))
            .when(hits_tp & hits_sl)
                .then(pl.lit(1, dtype=pl.Int8))           # Ambiguidade → NEUTRAL
            .when(hits_tp)
                .then(pl.lit(2, dtype=pl.Int8))           # BUY
            .when(hits_sl)
                .then(pl.lit(0, dtype=pl.Int8))           # SELL
            .otherwise(pl.lit(1, dtype=pl.Int8))          # Time-Stop → NEUTRAL
            .alias("target")
        ])

    # Cleanup meta cols (incluindo índices de toque First Touch)
    df_final = df.drop([c for c in [
        "__log_ret__", "volatility_ewma",
        "fwd_max_high", "fwd_min_low", "fwd_close_at_stop",
        "barrier_up", "barrier_dn", "fwd_tp_idx", "fwd_sl_idx"
    ] if c in df.columns])

    
    # Drop de linhas finais onde não temos janela de lookahead completa
    # (Evita bias de fim de arquivo onde high/low rolling sao parciais)
    max_ts = df_final["ts"].max()
    horizon_ms = int(mins * 60 * 1000)
    df_final = df_final.filter(pl.col("ts") <= (max_ts - horizon_ms))

    n_kept = len(df_final)
    n_total = len(df)
    logger.info(f"Triple Barrier Labeling: {n_kept:,} amostras geradas | {n_total - n_kept:,} bordas podadas.")

    return df_final


def run_labelling():
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

        # Calculate workers for pytest (v9.6 Robust Affinity Detection)
        try:
            try:
                # os.sched_getaffinity(0) detects real vCPUs assigned to the process (respects VM/cgroups)
                cpu_count = len(os.sched_getaffinity(0))
            except (AttributeError, ImportError, NotImplementedError):
                cpu_count = os.cpu_count() or 1
                
            lab_cfg = config.get('pre_processing', {}).get('labelling', {})
            if lab_cfg.get('use_dynamic_workers', False):
                # RULE: Available vCPUs - 1
                pytest_workers = max(1, cpu_count - 1)
            else:
                # RULE: Fixed value from master_config (currently 7)
                pytest_workers = lab_cfg.get('max_workers', 7)
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
    # Audit Logs → RESULTADOS_.../AUDITORIA/LABELLING/
    import yaml as _yaml
    with open("src/cloud/base_model/configs/master_config.yaml") as _f:
        _cfg = _yaml.safe_load(_f)
    upload_audit_to_drive(
        local_dirs=[f"{get_logs_root(_cfg)}/labelling"],
        stage_name="LABELLING",
        config=_cfg,
    )

