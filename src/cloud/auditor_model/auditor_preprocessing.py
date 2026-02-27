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

logger = logging.getLogger(__name__)

def load_config():
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def calculate_context_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores de contexto e regime de mercado.
    Entrada: DataFrame pandas (indexado pelo tempo, ou ordenado).
    Espera-se ter as colunas 'close', 'high' (se não tiver, usar proxies do tensor ou micro_price),
    e 'log_volume'. Aqui usaremos a premissa de que os parquets brutos já possuem close, bid_0_p, etc.
    """
    # Garantir ordenação temporal caso não seja índice de tempo formal
    # O pipeline salva close e micro_price, usaremos close
    if 'close' not in df_pd.columns:
        logger.warning("'close' não encontrado! Tentando usar 'micro_price' ou aproximando via bid_0_p.")
        if 'bid_0_p' in df_pd.columns and 'ask_0_p' in df_pd.columns:
            df_pd['close'] = (df_pd['bid_0_p'] + df_pd['ask_0_p']) / 2.0
        else:
            raise ValueError("Não há preço base para calcular indicadores no DataFrame.")

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

    # 6. Volatilidade Histórica de 1h (60 períodos de 1min)
    log_ret = np.log(c / c.shift(1))
    df_pd['vol_1h'] = log_ret.rolling(window=60).std() * np.sqrt(60)

    # 7. Volume Features (Proxy log_volume / tick_count)
    if 'log_volume' in df_pd.columns:
        v = np.exp(df_pd['log_volume']) - 1 # reconstruct tick counts
    else:
        v = pd.Series(1.0, index=df_pd.index) # fallback

    # Vol Z-Score (rolling 60 min para normalizar na sessão)
    vol_mean = v.rolling(window=60).mean()
    vol_std = v.rolling(window=60).std()
    df_pd['vol_zscore_1h'] = np.where(vol_std > 0, (v - vol_mean) / vol_std, 0.0)

    # Delta de Volume 24h (1440 min)
    # Como nem todos os arquivos têm 24h perfeitas integradas linearmente, 
    # faremos um drift do volume. Usaremos sum(last 60) vs sum(prev 60) como aproximação robusta intra-dia, 
    # ou tentamos o shift(1440) se houver dados.
    if len(df_pd) > 1440:
        vol_24h_sum = v.rolling(window=1440).sum()
        df_pd['delta_vol_24h'] = vol_24h_sum.pct_change(fill_method=None)
    else:
        # Fallback short-term delta se os chunks de arquivo forem curtos
        logger.warning(f"⚠️ AVISO: Dados insuficientes para delta_vol_24h (Tamanho: {len(df_pd)} < 1440). Usando fallback de 4h. No ambiente live, o aquecimento (warm-up) via REST API com 12h+ mitiga este drift.")
        vol_4h_sum = v.rolling(window=240).sum()
        df_pd['delta_vol_24h'] = vol_4h_sum.pct_change(fill_method=None)

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
    
    # Dynamic window size falling back if len < 1440
    window_vwap = min(1440, len(df_pd)) if len(df_pd) > 0 else 1
    
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

    logger.info(f"Processando {len(parquet_files)} arquivos em {input_dir}...")
    
    for pf in parquet_files:
        try:
            df = pl.read_parquet(pf).to_pandas()
            df_enriched = calculate_context_features(df)
            
            # Manter apenas as colunas essenciais para o output, economizando espaço
            core_features = [
                'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
                'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
                'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
            ]
            
            # Se você precisar manter os dados do livro, pode exportar tudo. 
            # Mas aqui o auditor precisará da concatenação. Vamos manter o target histórico também.
            export_cols = core_features
            if 'target' in df_enriched.columns:
                export_cols.append('target')
                
            # Converter de volta para polars
            df_out = pl.DataFrame(df_enriched[export_cols])
            out_file = output_path / f"context_{pf.name}"
            df_out.write_parquet(out_file)
            logger.info(f"   [OK] Processado e salvo: {out_file.name}")
        except Exception as e:
            logger.error(f"Erro processando {pf.name}: {e}")

if __name__ == "__main__":
    setup_logger("auditor_preprocessing", "")
    config = load_config()
    
    # ── Resolve Paths ──────────────────────────────────────────────────────────
    # O user especificou que o auditor usara como dataset de treino o VAL de SPECIALIZED
    # (Strict OOF) para gerar o contexto neutro e seguro.
    
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    base_labelled_name = f"labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    spec_val_dir = Path(f"data/L2/splits_specialized_{base_labelled_name}/val")
    
    out_context_dir = Path("data/auditor/context")
    
    if spec_val_dir.exists():
        logger.info(f"Processando contexto do Auditor a partir dos dados OOF: {spec_val_dir}")
        process_and_save_context(spec_val_dir, out_context_dir)
    else:
        logger.error(f"❌ Nao encontrou {spec_val_dir}. Rode o split_dataset.py primeiro.")
        
    logger.info("✅ Auditor Preprocessing (Context Engineering) Finalizado OOF.")
