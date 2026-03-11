import os
import sys
from pathlib import Path
import yaml
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.path_utils import get_labelled_dir

def load_config():
    with open("src/cloud/base_model/configs/master_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_visual_audit():
    config = load_config()
    
    # 1. Parâmetros de Tempo
    res_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
    res_min = int("".join(filter(str.isdigit, res_freq)) or "1")
    
    # Lookback in logic bars directly from config instead of lookback_minutes
    lookback_bars = config['optimization'].get('seq_len', 60)
    
    lookahead_min = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    lookahead_bars = lookahead_min // res_min

    # 2. Localizar Arquivo Rotulado
    labelled_dir = Path(get_labelled_dir(config))
    # tentar pegar na raiz ou dentro de train
    parquet_files = list(labelled_dir.glob("*.parquet")) + list(labelled_dir.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ Nenhum arquivo parquet encontrado em {labelled_dir} ou subpastas.")
        return
        
    # Pega o primeiro como amostra
    sample_file = parquet_files[0]
    print(f"📂 Usando arquivo de amostra: {sample_file}")
    
    df = pl.read_parquet(sample_file)
    print(f"📊 Dataset carregado com {len(df)} linhas.")
    
    # 3. Preparar a feature de "Preço"
    # Se temos close_price ou algo parecido:
    price_col = None
    for col in ['close_price', 'close', 'mid_price']:
        if col in df.columns:
            price_col = col
            break
            
    if not price_col:
        # Tentar usar spread cumulativo ou outra coisa direcional
        # Se só temos os logs de retornos normalizados, vamos usar cumsum do 'close_returns' se existir
        print("⚠️ Feature de preço bruto não encontrada diretamente.")
        cols = df.columns
        log_ret_col = next((c for c in cols if 'return' in c.lower() or 'log' in c.lower()), None)
        if log_ret_col:
            print(f"🔄 Reconstruindo trajetória baseada na feature: {log_ret_col}")
            price_col = "pseudo_price"
            df = df.with_columns(pl.col(log_ret_col).cum_sum().alias(price_col))
        else:
            # Fallback total, usar a primeira feature só pra ter uma linha
            price_col = cols[0]
            if price_col == 'timestamp': price_col = cols[1]
            print(f"⚠️ Usando feature arbitrária: {price_col}")

    # Checar coluna the target
    target_col = 'target'
    if target_col not in df.columns:
         target_col = next((c for c in df.columns if 'label' in c.lower() or 'target' in c.lower()), None)
    
    if not target_col:
        print(f"❌ Coluna de label não encontrada no parquet.")
        return

    # Drop null targets to ensure we only plot valid predictions
    df = df.drop_nulls(subset=[target_col])

    # 4. Amostrar índices homogeneamente
    # Evitar pontas porque precisamos ter espaço para o lookback e para o lookahead!
    valid_start = lookback_bars
    valid_end = len(df) - lookahead_bars - 1
    
    if valid_start >= valid_end:
        print(f"❌ Dataset muito pequeno ({len(df)} linhas) para suportar lookback {lookback_bars} e lookahead {lookahead_bars}")
        return
    indices_to_plot = np.linspace(valid_start, valid_end, 40, dtype=int)
    # 5. Configurar pasta de saída
    out_dir = Path("PRINT_VAL")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract as numpy for easy slicing
    prices = df[price_col].to_numpy()
    highs = df['high'].to_numpy() if 'high' in df.columns else prices
    lows = df['low'].to_numpy() if 'low' in df.columns else prices
    labels = df[target_col].to_numpy()
    # If there's a timestamp
    timestamps = None
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].to_numpy()

    # Thresold values
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)


    # Dicionário de Cores e Nomes
    target_map = {
        0: {"name": "SELL", "color": "red"},
        1: {"name": "NEUTRAL", "color": "gray"},
        2: {"name": "BUY", "color": "green"}
    }

    # 6. Plotar as imagens
    print(f"📸 Gerando {len(indices_to_plot)} previsões...")
    
    for count, idx in enumerate(indices_to_plot, 1):
        target_val = int(labels[idx])
        conf = target_map.get(target_val, {"name": f"UNKNOWN_{target_val}", "color": "blue"})
        
        # Puxar X e Y das fatias
        x_lookback = np.arange(idx - lookback_bars + 1, idx + 1)
        y_lookback = prices[idx - lookback_bars + 1 : idx + 1]
        
        x_lookahead = np.arange(idx, idx + lookahead_bars + 1)
        y_lookahead = prices[idx : idx + lookahead_bars + 1]
        h_lookahead = highs[idx : idx + lookahead_bars + 1]
        l_lookahead = lows[idx : idx + lookahead_bars + 1]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Limites Target
        now_price = prices[idx]
        target_buy = now_price * (1 + buy_th)
        target_sell = now_price * (1 - sell_th)

        # Draw bounds
        ax.axhline(y=target_buy, color='green', linestyle=':', alpha=0.7, label=f'+{buy_th*100:.2f}% (BUY Trigger)')
        ax.axhline(y=target_sell, color='red', linestyle=':', alpha=0.7, label=f'-{sell_th*100:.2f}% (SELL Trigger)')
        
        # Linha base do lookback
        ax.plot(x_lookback, y_lookback, color='black', label=f'Lookback ({lookback_min}m)')
        
        # Ranges High/Low do Lookahead
        ax.fill_between(x_lookahead, l_lookahead, h_lookahead, color=conf["color"], alpha=0.2, label='High/Low Range')
        
        # Linha projetada do Lookahead (Close)
        ax.plot(x_lookahead, y_lookahead, color=conf["color"], linestyle='--', linewidth=2, label=f'Lookahead Close')
        
        # Ponto exato de "NOW"
        ax.scatter(idx, prices[idx], color='blue', s=100, zorder=5, label='NOW')
        # Ponto futuro (alvo da marcação)
        ax.scatter(idx + lookahead_bars, prices[idx + lookahead_bars], facecolors='none', edgecolors=conf["color"], s=100, zorder=5)
        
        # Draw horizontal baseline at "NOW" price for visual reference
        ax.axhline(y=prices[idx], color='gray', linestyle=':', alpha=0.5)
        
        # Title handling with timestamp if available
        ts_str = f"Ts_idx={idx}"
        if timestamps is not None:
             # Just roughly stringify
             ts_str = str(timestamps[idx]).replace('T', ' ')[:19]
             
        ax.set_title(f"Label: {conf['name']} ({target_val}) | {ts_str}", fontsize=14, fontweight='bold', color=conf["color"])
        ax.set_ylabel(price_col)
        ax.set_xlabel('Index (Bars)')
        ax.grid(alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
        
        out_name = out_dir / f"print_{count:02d}_{conf['name']}.png"
        fig.savefig(out_name, dpi=120)
        plt.close(fig)
        
    print(f"✅ {len(indices_to_plot)} prints salvos na pasta 'PRINT_VAL/'.")


if __name__ == "__main__":
    run_visual_audit()
