
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

def generate_label_plots():
    labelled_dir = Path("data/L2/labelled")
    output_dir = Path("data/artifacts/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    files = list(labelled_dir.glob("*.parquet"))
    if not files:
        print("Erro: Nenhum dado rotulado encontrado.")
        return
    
    # Load raw data to have access to full prices for calculation
    df = pl.read_parquet(files[0]) 
    print(f"Lendo {files[0].name} para visualização...")
    
    lookahead = 60
    window_before = 60
    
    labels = {0: "Sell", 1: "Neutral", 2: "Buy"}
    colors = {0: 'tab:red', 1: 'tab:gray', 2: 'tab:green'}
    
    for label_id, label_name in labels.items():
        print(f"Gerando 6 plots para Label {label_id} ({label_name})...")
        
        # Encontrar todos os índices com este label
        indices = df.with_row_index().filter(
            (pl.col("target") == label_id) & 
            (pl.col("index") > window_before) & 
            (pl.col("index") < len(df) - lookahead - 1)
        ).select("index").to_numpy().flatten()
        
        if len(indices) == 0:
            print(f"Aviso: Nenhum exemplo encontrado para {label_name}.")
            continue
            
        count = min(len(indices), 6)
        samples = random.sample(list(indices), count)
            
        for i, idx in enumerate(samples):
            # Janela de dados: 60min antes + 60min depois (lookahead)
            # Precisamos de window_before + 1 + lookahead pontos
            start = idx - window_before
            length = window_before + lookahead + 1
            
            chunk = df.slice(start, length)
            prices = chunk["close"].to_numpy()
            
            plt.figure(figsize=(12, 6))
            
            # X-axis relative to decision point (0)
            x = np.arange(-window_before, lookahead + 1)
            
            # Plot historical (up to 0)
            plt.plot(x[:window_before+1], prices[:window_before+1], color='black', alpha=0.7, label='Histórico')
            
            # Plot future (starting from 0)
            plt.plot(x[window_before:], prices[window_before:], 
                     color=colors[label_id], linewidth=2.5, label=f'Lookahead ({label_name})')
            
            # Decision point marker
            plt.scatter(0, prices[window_before], color='gold', zorder=5, s=150, edgecolors='black', label='Trigger')
            
            # Horizontal line at decision price for visual reference
            plt.axhline(y=prices[window_before], color='gray', linestyle='--', alpha=0.4)
            
            # Return calculation (matches labeling logic)
            p_now = prices[window_before]
            p_future = prices[-1]
            ret_pct = (p_future / p_now - 1) * 100
            
            plt.title(f"Label {label_id} ({label_name}) | Return: {ret_pct:+.2f}% | Index: {idx}", fontsize=14)
            plt.xlabel("Minutos (Relativo ao Trigger)", fontsize=12)
            plt.ylabel("Preço BTC", fontsize=12)
            plt.legend(loc='upper left')
            plt.grid(True, linestyle=':', alpha=0.6)
            
            plt.savefig(output_dir / f"label_{label_id}_sample_{i}.png", dpi=100)
            plt.close()

if __name__ == "__main__":
    generate_label_plots()
