import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
import logging
import sys
import yaml

project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger
from src.cloud.base_model.utils.path_utils import get_auditor_context_dir, get_labelled_dir

logger = logging.getLogger(__name__)

def load_config():
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def calculate_context_features_polars(lf: pl.LazyFrame, resample_min: int = 1) -> pl.LazyFrame:
    """
    Refatoração v10.5: Puro Polars Lazy. RAM-Efficient.
    Calcula indicadores Alpha Sensors sem sair do Polars.
    """
    bars_per_hour = max(1, 60  // resample_min)
    bars_per_day  = max(1, 1440 // resample_min)

    # 1. Base Price Logic
    # Se 'close' não existe, tenta micro_price ou bid/ask
    lf = lf.with_columns([
        pl.when(pl.col("close").is_null())
          .then(pl.coalesce(["micro_price", (pl.col("bid_0_p") + pl.col("ask_0_p")) / 2.0]))
          .otherwise(pl.col("close"))
          .alias("close")
    ])

    # Pseudo-High/Low if missing
    lf = lf.with_columns([
        pl.when(pl.col("high").is_null())
          .then(pl.col("close") + pl.coalesce(["volatility", pl.col("close") * 0.001]))
          .otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("low").is_null())
          .then(pl.col("close") - pl.coalesce(["volatility", pl.col("close") * 0.001]))
          .otherwise(pl.col("low")).alias("low")
    ])

    # 2. Indicators using Polars Expressions
    lf = lf.with_columns([
        # Trend
        pl.col("close").ewm_mean(span=8, adjust=False).alias("ema_8"),
        pl.col("close").ewm_mean(span=21, adjust=False).alias("ema_21"),
        
        # Bollinger
        pl.col("close").rolling_mean(window_size=20).alias("bb_mean"),
        pl.col("close").rolling_std(window_size=20).alias("bb_std"),
        
        # RSI components
        (pl.col("close") - pl.col("close").shift(1)).alias("delta")
    ])

    lf = lf.with_columns([
        pl.col("delta").where(pl.col("delta") > 0).fill_null(0).alias("gain"),
        pl.col("delta").where(pl.col("delta") < 0).fill_null(0).abs().alias("loss"),
        pl.when(pl.col("ema_8") > pl.col("ema_21")).then(1).otherwise(-1).alias("ema_trend"),
        ((pl.col("ema_8") - pl.col("ema_21")) / pl.col("ema_21")).alias("ema_cross_dist"),
        (pl.col("bb_mean") + 2 * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("bb_mean") - 2 * pl.col("bb_std")).alias("bb_lower"),
    ])

    lf = lf.with_columns([
        (100 - (100 / (1 + (pl.col("gain").rolling_mean(14) / (pl.col("loss").rolling_mean(14) + 1e-9))))).alias("rsi_14"),
        (100 * (pl.col("close") - pl.col("low").rolling_min(14)) / (pl.col("high").rolling_max(14) - pl.col("low").rolling_min(14) + 1e-9)).alias("stoch_14"),
        # True Range
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        ]).alias("tr")
    ])

    lf = lf.with_columns([
        pl.col("tr").rolling_mean(14).alias("atr_14"),
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret"),
        pl.when((pl.col("bb_upper") - pl.col("bb_lower")) > 0)
          .then((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower")))
          .otherwise(0.5).alias("bb_pct")
    ])

    lf = lf.with_columns([
        (pl.col("atr_14") / pl.col("close")).alias("atr_norm"),
        (pl.col("log_ret").rolling_std(bars_per_hour) * np.sqrt(bars_per_hour)).alias("vol_1h")
    ])

    # Volume Indicators
    if "log_volume" in lf.columns:
        lf = lf.with_columns(pl.col("log_volume").exp().alias("v"))
    else:
        lf = lf.with_columns(pl.lit(1.0).alias("v"))

    lf = lf.with_columns([
        pl.col("v").rolling_mean(bars_per_hour).alias("v_mean"),
        pl.col("v").rolling_std(bars_per_hour).alias("v_std"),
        pl.col("v").rolling_sum(bars_per_day).alias("v_sum_day")
    ])

    lf = lf.with_columns([
        pl.when(pl.col("v_std") > 0).then((pl.col("v") - pl.col("v_mean")) / pl.col("v_std")).otherwise(0.0).alias("vol_zscore_1h"),
        (pl.col("v_sum_day") / pl.col("v_sum_day").shift(1) - 1).alias("delta_vol_24h")
    ])

    # ADX Logic (Wilder's Smoothing via ewm_mean alpha=1/14)
    lf = lf.with_columns([
        (pl.col("high") - pl.col("high").shift(1)).alias("up_m"),
        (pl.col("low").shift(1) - pl.col("low")).alias("dn_m")
    ])
    lf = lf.with_columns([
        pl.when((pl.col("up_m") > pl.col("dn_m")) & (pl.col("up_m") > 0)).then(pl.col("up_m")).otherwise(0.0).alias("p_dm"),
        pl.when((pl.col("dn_m") > pl.col("up_m")) & (pl.col("dn_m") > 0)).then(pl.col("dn_m")).otherwise(0.0).alias("n_dm")
    ])
    lf = lf.with_columns([
        (100 * (pl.col("p_dm").ewm_mean(alpha=1/14, adjust=False) / (pl.col("atr_14") + 1e-9))).alias("p_di"),
        (100 * (pl.col("n_dm").ewm_mean(alpha=1/14, adjust=False) / (pl.col("atr_14") + 1e-9))).alias("n_di")
    ])
    lf = lf.with_columns([
        (100 * (pl.col("p_di") - pl.col("n_di")).abs() / (pl.col("p_di") + pl.col("n_di") + 1e-9)).alias("dx")
    ])
    lf = lf.with_columns(pl.col("dx").ewm_mean(alpha=1/14, adjust=False).alias("adx_14"))

    # VWAP and MFI
    lf = lf.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("tp"),
    ])
    lf = lf.with_columns([
        (pl.col("tp") * pl.col("v")).alias("tpv"),
        pl.col("tp").rolling_std(window_size=bars_per_day).alias("tp_std")
    ])
    lf = lf.with_columns([
        (pl.col("tpv").rolling_sum(bars_per_day) / (pl.col("v").rolling_sum(bars_per_day) + 1e-9)).alias("vwap_rolling")
    ])
    lf = lf.with_columns([
        pl.when(pl.col("tp_std") > 0).then((pl.col("close") - pl.col("vwap_rolling")) / pl.col("tp_std")).otherwise(0.0).alias("vwap_zscore")
    ])

    # MFI
    lf = lf.with_columns([
        pl.when(pl.col("tp") > pl.col("tp").shift(1)).then(pl.col("tpv")).otherwise(0.0).alias("pos_f"),
        pl.when(pl.col("tp") < pl.col("tp").shift(1)).then(pl.col("tpv")).otherwise(0.0).alias("neg_f")
    ])
    lf = lf.with_columns([
        (100 - (100 / (1 + (pl.col("pos_f").rolling_sum(14) / (pl.col("neg_f").rolling_sum(14) + 1e-9))))).alias("mfi_14")
    ])

    # Book Skew Proxy (if ETL features present)
    if "book_skew_bid" not in lf.columns:
        lf = lf.with_columns(pl.lit(0.0).alias("book_skew_bid"))
    if "book_skew_ask" not in lf.columns:
        lf = lf.with_columns(pl.lit(0.0).alias("book_skew_ask"))

    # Clean up and final fill
    final_sensors = [
        'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
        'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
        'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
    ]
    
    # Fill nulls (Polars equivalent of ffill().bfill())
    for s in final_sensors:
        lf = lf.with_columns(pl.col(s).forward_fill().backward_fill().fill_null(0.0))

    return lf


def process_and_save_context(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(list(input_path.glob("*.parquet")))
    if not parquet_files:
        logger.error(f"Nenhum arquivo encontrado em {input_dir}")
        return

    logger.info(f"Carregando {len(parquet_files)} arquivos para processamento CONTÍNUO (PURE POLARS)...")

    config = load_config()
    resample_freq = config.get('pre_processing', {}).get('etl', {}).get('resample_freq', '1min')
    resample_min = max(1, int(pd.to_timedelta(resample_freq).total_seconds() // 60))
    logger.info(f"auditor_preprocessing: resample_freq={resample_freq} ({resample_min} min/bar)")

    # v10.5: Uso de LazyFrame para economia radical de RAM
    needed_cols = [
        'close', 'micro_price', 'bid_0_p', 'ask_0_p', 'volatility', 
        'log_volume', 'book_skew_bid', 'book_skew_ask', 'target'
    ]
    
    # 1. Escanear todos os arquivos e UNIFICAR SCHEMA
    lfs = []
    for pf in parquet_files:
        try:
            # v10.7: Scan filtrando colunas imediatamente para reduzir o plano
            lf_i = pl.scan_parquet(pf)
            
            # Garante schema limpo e uniforme
            exprs = []
            available = lf_i.columns
            for c in needed_cols:
                if c in available:
                    exprs.append(pl.col(c).cast(pl.Float64))
                else:
                    exprs.append(pl.lit(None).cast(pl.Float64).alias(c))
            
            # Adiciona filename e executa a seleção
            lf_i = lf_i.select(exprs).with_columns(pl.lit(pf.name).alias("_filename"))
            lfs.append(lf_i)
        except Exception as e:
            logger.warning(f"Erro ao escanear {pf.name}: {e}")

    if not lfs:
        logger.error("Nenhum dado válido encontrado.")
        return

    # v10.7: Uso de vertical concat explícito
    lf_full = pl.concat(lfs, how="vertical")
    
    # 2. Calcular indicadores em modo LAZY
    logger.info("Calculando indicators Alpha Sensors em modo Lazy...")
    lf_enriched = calculate_context_features_polars(lf_full, resample_min=resample_min)

    # 3. Realizar o cálculo (Collect)
    logger.info(f"Executando plano de cálculo (Collect + STREAMING) para {len(parquet_files)} arquivos...")
    try:
        # v10.7: STREAMING=True é a chave para o Polars não engasgar em planos complexos
        df_result = lf_enriched.collect(streaming=True)
    except Exception as e:
        logger.error(f"❌ Falha crítica no Colect/Streaming: {e}")
        raise
    
    # 4. Salvar cada fragmento de volta
    logger.info(f"Desmembrando e salvando em {output_dir}...")
    final_sensors = [
        'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
        'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
        'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
    ]
    export_cols = final_sensors + (['target'] if 'target' in df_result.columns else [])
    
    for filename in [pf.name for pf in parquet_files]:
        df_slice = df_result.filter(pl.col("_filename") == filename).select(export_cols)
        out_file = output_path / f"context_{filename}"
        df_slice.write_parquet(out_file)
    
    logger.info(f"✅ Processamento Polars-Lazy finalizado para {len(parquet_files)} arquivos.")

if __name__ == "__main__":
    setup_logger("auditor_preprocessing", "")
    config = load_config()

    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    base_labelled_name = f"labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")

    # ── Fix (Audit v4.9): usar Foundation Val como source.
    # O K-Fold OOF cobre o Foundation Val (splits_*/val). Para que o inner join
    # em auditor_labelling.py por `original_row_idx` seja válido, as context
    # features precisam indexar o mesmo conjunto de linhas. Usar spec_val (20%
    # do val) criava um espaço de índices incompatível — os índices seriam
    # 0..N_spec mas o OOF esperaria 0..N_foundation_val. Fix: sempre usar
    # Foundation Val como fonte de context features.
    foundation_val_dir = Path(get_labelled_dir(config)) / "val"

    out_context_dir = Path(get_auditor_context_dir(config))

    if foundation_val_dir.exists():
        logger.info(f"📂 [Audit Fix] Context features extraídas de Foundation Val: {foundation_val_dir}")
        process_and_save_context(foundation_val_dir, out_context_dir)
    else:
        logger.error(
            f"❌ Foundation Val não encontrado: {foundation_val_dir}. "
            "Rode split_dataset.py primeiro."
        )

    logger.info("✅ Auditor Preprocessing (Context Engineering) Finalizado OOF.")
