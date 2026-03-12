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

def calculate_context_features(df_pd: pd.DataFrame, resample_min: int = 1) -> pd.DataFrame:
    """
    Calcula indicadores de contexto e regime de mercado.
    Entrada: DataFrame pandas (indexado pelo tempo, ou ordenado).
    Espera-se ter as colunas 'close', 'high' (se não tiver, usar proxies do tensor ou micro_price),
    e 'log_volume'. Aqui usaremos a premissa de que os parquets brutos já possuem close, bid_0_p, etc.

    Args:
        df_pd:        Input DataFrame.
        resample_min: Bar duration in real minutes (e.g. 5 for 5min bars). Used to compute all
                      rolling windows dynamically so that economic meaning is preserved regardless
                      of temporal resolution. Defaults to 1 (1min bars, legacy behaviour).
    """
    # v4.9: Dynamic window sizes — all windows expressed in BARS, not hardcoded minutes
    bars_per_hour = max(1, 60  // resample_min)   # e.g. 12 bars @ 5min, 60 bars @ 1min
    bars_per_day  = max(1, 1440 // resample_min)  # e.g. 288 bars @ 5min, 1440 bars @ 1min
    bars_4h       = max(1, 240  // resample_min)  # Fallback for short DataFrames

    if 'close' not in df_pd.columns:
        if 'micro_price' in df_pd.columns:
            df_pd['close'] = df_pd['micro_price']
        elif 'bid_0_p' in df_pd.columns and 'ask_0_p' in df_pd.columns:
            df_pd['close'] = (df_pd['bid_0_p'] + df_pd['ask_0_p']) / 2.0
        else:
            raise ValueError("Não há preço base ('close', 'micro_price' ou 'bid_0_p') para calcular indicadores no DataFrame.")

    # High/Low pseudo-rebuy logic (se não disponíveis, simulamos a volatilidade)
    # Como os dados de input têm 'volatility' (std do micro_price), podemos aproximar High e Low
    if 'volatility' in df_pd.columns:
        df_pd['high'] = df_pd['close'] + df_pd['volatility']
        df_pd['low'] = df_pd['close'] - df_pd['volatility']
    else:
        df_pd['high'] = df_pd['close'] * 1.001
        df_pd['low'] = df_pd['close'] * 0.999

    c = df_pd['close']
    h = df_pd['high']
    l = df_pd['low']

    # 1. EMA 8 e EMA 21 (Trend)
    df_pd['ema_8'] = c.ewm(span=8, adjust=False).mean()
    df_pd['ema_21'] = c.ewm(span=21, adjust=False).mean()
    df_pd['ema_trend'] = np.where(df_pd['ema_8'] > df_pd['ema_21'], 1, -1)
    df_pd['ema_cross_dist'] = (df_pd['ema_8'] - df_pd['ema_21']) / df_pd['ema_21']

    # 2. Bollinger Bands (20, std=2)
    rolling_mean = c.rolling(window=20).mean()
    rolling_std = c.rolling(window=20).std()
    df_pd['bb_upper'] = rolling_mean + (rolling_std * 2)
    df_pd['bb_lower'] = rolling_mean - (rolling_std * 2)
    # Posição do preço (%B)
    bb_range = df_pd['bb_upper'] - df_pd['bb_lower']
    df_pd['bb_pct'] = np.where(bb_range > 0, (c - df_pd['bb_lower']) / bb_range, 0.5)

    # 3. RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df_pd['rsi_14'] = 100 - (100 / (1 + rs))

    # 4. Estocástico (14)
    lowest_low = l.rolling(window=14).min()
    highest_high = h.rolling(window=14).max()
    stoch_range = highest_high - lowest_low
    df_pd['stoch_14'] = np.where(stoch_range > 0, 100 * ((c - lowest_low) / stoch_range), 50.0)

    # 5. ATR (14)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_pd['atr_14'] = tr.rolling(window=14).mean()
    df_pd['atr_norm'] = df_pd['atr_14'] / c

    # 6. Volatilidade Histórica de 1h (bars_per_hour períodos)
    log_ret = np.log(c / c.shift(1))
    df_pd['vol_1h'] = log_ret.rolling(window=bars_per_hour).std() * np.sqrt(bars_per_hour)

    # v4.9: np.expm1 is numerically more accurate than np.exp()-1 near zero
    if 'log_volume' in df_pd.columns:
        v = np.expm1(df_pd['log_volume'])  # inverse of log1p(tick_count)
    else:
        v = pd.Series(1.0, index=df_pd.index) # fallback

    # Vol Z-Score (rolling bars_per_hour for intra-session normalization)
    vol_mean = v.rolling(window=bars_per_hour).mean()
    vol_std  = v.rolling(window=bars_per_hour).std()
    df_pd['vol_zscore_1h'] = np.where(vol_std > 0, (v - vol_mean) / vol_std, 0.0)

    # v4.9.1: Dynamic Growing Window for Delta Vol (requested by user)
    # Instead of a hard fallback of 4h, we use the MAX available history up to 288 bars.
    actual_bars = len(df_pd)
    if actual_bars < bars_per_day:
        # Progress logging with visual prominence
        from src.cloud.base_model.utils.color_utils import TerminalColors as TC
        progress_pct = (actual_bars / bars_per_day) * 100
        msg = f"Auditor 24h Sensor: {actual_bars}/{bars_per_day} bars ({progress_pct:.1f}% filled)"
        
        logger.info("######################################################")
        logger.info(f"# {TC.color_text(msg, TC.CYAN)}")
        logger.info("######################################################")
        
        # Use whatever we have (min 1 bar to avoid division by zero)
        dynamic_window = max(2, actual_bars - 1) 
        vol_sum = v.rolling(window=dynamic_window, min_periods=1).sum()
        df_pd['delta_vol_24h'] = vol_sum.pct_change(fill_method=None)
    else:
        # Full 24h window
        vol_day_sum = v.rolling(window=bars_per_day).sum()
        df_pd['delta_vol_24h'] = vol_day_sum.pct_change(fill_method=None)

    # ── Refatoração v4.2: Alpha Sensors ──────────────────────────────────────
    
    # 8. ADX (Average Directional Index - 14)
    # Wilder's Smoothing: aprox via ewm(alpha=1/14)
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    pos_dm_ser = pd.Series(pos_dm, index=df_pd.index)
    neg_dm_ser = pd.Series(neg_dm, index=df_pd.index)
    
    smoothed_pos_dm = pos_dm_ser.ewm(alpha=1/14, adjust=False).mean()
    smoothed_neg_dm = neg_dm_ser.ewm(alpha=1/14, adjust=False).mean()
    smoothed_tr = df_pd['atr_14'] # Já temos o ATR(14) mapeado
    
    # +DI e -DI
    plus_di = 100 * (smoothed_pos_dm / (smoothed_tr + 1e-9))
    minus_di = 100 * (smoothed_neg_dm / (smoothed_tr + 1e-9))
    
    # DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df_pd['adx_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

    # 9. VWAP Z-Score (1440 Janela Móvel / 24h)
    typical_price = (h + l + c) / 3.0
    vp = typical_price * v
    
    # Dynamic window size falling back if len < bars_per_day
    window_vwap = min(bars_per_day, len(df_pd)) if len(df_pd) > 0 else 1
    
    vwap_rolling = vp.rolling(window=window_vwap, min_periods=1).sum() / (v.rolling(window=window_vwap, min_periods=1).sum() + 1e-9)
    vwap_std = typical_price.rolling(window=window_vwap, min_periods=1).std()
    
    df_pd['vwap_zscore'] = np.where(vwap_std > 0, (c - vwap_rolling) / vwap_std, 0.0)

    # 10. MFI (Money Flow Index - 14)
    raw_money_flow = typical_price * v
    
    pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0.0)
    neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0.0)
    
    pos_flow_ser = pd.Series(pos_flow, index=df_pd.index)
    neg_flow_ser = pd.Series(neg_flow, index=df_pd.index)
    
    pos_flow_14 = pos_flow_ser.rolling(window=14, min_periods=1).sum()
    neg_flow_14 = neg_flow_ser.rolling(window=14, min_periods=1).sum()
    
    mfi_ratio = pos_flow_14 / (neg_flow_14 + 1e-9)
    df_pd['mfi_14'] = 100 - (100 / (1 + mfi_ratio))

    # 11. Book Skewness (Assimetria L200 bidirecional)
    # Procurar por max níveis de bid e ask para derivar a cauda
    bid_cols = [col for col in df_pd.columns if col.startswith('bid_') and col.endswith('_s')]
    ask_cols = [col for col in df_pd.columns if col.startswith('ask_') and col.endswith('_s')]
    
    if bid_cols:
        # df_pd[bid_cols].skew(axis=1) is Fisher-Pearson coefficient of skewness for each row
        df_pd['book_skew_bid'] = df_pd[bid_cols].skew(axis=1)
    else:
        df_pd['book_skew_bid'] = 0.0
        
    if ask_cols:
        df_pd['book_skew_ask'] = df_pd[ask_cols].skew(axis=1)
    else:
        df_pd['book_skew_ask'] = 0.0

    # Preencher NaNs com fill forward e bfill (para o início da série)
    cols_to_fill = [
        'ema_8', 'ema_21', 'ema_trend', 'ema_cross_dist', 'bb_upper', 'bb_lower', 'bb_pct',
        'rsi_14', 'stoch_14', 'atr_14', 'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
        'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
    ]
    df_pd[cols_to_fill] = df_pd[cols_to_fill].ffill().bfill().fillna(0.0)

    return df_pd

def process_and_save_context(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(list(input_path.glob("*.parquet")))
    if not parquet_files:
        logger.error(f"Nenhum arquivo encontrado em {input_dir}")
        return

    logger.info(f"Carregando {len(parquet_files)} arquivos para processamento CONTÍNUO...")

    # v4.9: resolve resample_min from master_config
    config = load_config()
    resample_freq = config.get('pre_processing', {}).get('etl', {}).get('resample_freq', '1min')
    import pandas as _pd_inner
    resample_min = max(1, int(_pd_inner.to_timedelta(resample_freq).total_seconds() // 60))
    logger.info(f"auditor_preprocessing: resample_freq={resample_freq} ({resample_min} min/bar)")

    # 1. Carregar todos os arquivos e manter o controle de tamanhos
    all_dfs = []
    file_metadata = [] # list of (filename, row_count)

    for pf in parquet_files:
        try:
            df_i = pl.read_parquet(pf)
            row_count = len(df_i)
            all_dfs.append(df_i)
            file_metadata.append((pf.name, row_count))
        except Exception as e:
            logger.error(f"Erro ao carregar {pf.name}: {e}")

    if not all_dfs:
        return

    # 2. Concatenar em uma série temporal única e contínua
    logger.info(f"Concatenando {len(all_dfs)} dataframes para cálculo global de indicadores...")
    df_full = pl.concat(all_dfs).to_pandas()
    
    # 3. Calcular indicadores sobre a série completa (SEM BURACOS NAS FRONTEIRAS)
    logger.info("Calculando indicators Alpha Sensors sobre o dataset completo...")
    df_full_enriched = calculate_context_features(df_full, resample_min=resample_min)

    # 4. Definir colunas de exportação
    core_features = [
        'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
        'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
        'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
    ]
    export_cols = core_features
    if 'target' in df_full_enriched.columns:
        export_cols.append('target')

    # 5. Splitter: Quebrar de volta nos arquivos originais e salvar
    logger.info(f"Desmembrando {len(file_metadata)} arquivos e salvando em {output_dir}...")
    current_start = 0
    for filename, row_count in file_metadata:
        df_slice = df_full_enriched.iloc[current_start : current_start + row_count]
        
        # Converter de volta para polars
        df_out = pl.DataFrame(df_slice[export_cols])
        out_file = output_path / f"context_{filename}"
        df_out.write_parquet(out_file)
        
        current_start += row_count
        # logger.info(f"   [OK] Processado e salvo: {out_file.name}")

    logger.info(f"✅ Processamento contínuo finalizado para {len(file_metadata)} arquivos.")

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
