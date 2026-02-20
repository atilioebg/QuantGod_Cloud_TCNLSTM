import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_validation_plots():
    output_plot_dir = Path("plots/label_check")
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path("data/L2/labelled_test_SELL_0020_BUY_0020_1h")
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return
        
    files = sorted(list(input_dir.glob("*.parquet")))
    colors = {0: 'red', 1: 'gray', 2: 'green'}
    labels = {0: 'SELL', 1: 'NEUTRAL', 2: 'BUY'}
    
    for target_class in [0, 1, 2]:
        print(f"Generating plots for {labels[target_class]}...")
        count = 0
        for f in files:
            df = pl.read_parquet(f)
            # Find indices where target matches
            indices = []
            for i, val in enumerate(df['target'].to_list()):
                if val == target_class:
                    indices.append(i)
            
            if not indices:
                continue
            
            # Pick a few diverse indices (not all from the start)
            selected_indices = [indices[len(indices)//4], indices[len(indices)//2], indices[3*len(indices)//4], indices[-1]]
            if len(indices) < 4:
                selected_indices = indices
            
            for idx in selected_indices:
                # Window: 1h before, 2h after (lookahead is 1h)
                start = max(0, idx - 60)
                end = min(len(df), idx + 180)
                window = df.slice(start, end - start)
                
                plt.figure(figsize=(12, 6))
                prices = window['close'].to_list()
                plt.plot(prices, color='black', alpha=0.4, label='Price')
                
                # Highlight the target point
                target_idx_in_window = idx - start
                plt.scatter(target_idx_in_window, prices[target_idx_in_window], color=colors[target_class], s=120, label=labels[target_class], zorder=5)
                
                # Mark the lookahead point (1h later)
                lookahead_idx = target_idx_in_window + 60
                if lookahead_idx < len(prices):
                    plt.axvline(x=lookahead_idx, color='blue', linestyle='--', alpha=0.5, label='1h Lookahead')
                    plt.scatter(lookahead_idx, prices[lookahead_idx], color='blue', s=80, marker='x', alpha=0.8)
                    
                    # Calculate return for label verification
                    ret = (prices[lookahead_idx] / prices[target_idx_in_window]) - 1
                    plt.title(f"{labels[target_class]} | File: {f.name} | Return: {ret:.4%}")
                else:
                    plt.title(f"{labels[target_class]} | File: {f.name} | (End of file)")

                plt.xlabel("Minutes")
                plt.ylabel("Micro Price")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.savefig(output_plot_dir / f"{labels[target_class]}_{count}.png", dpi=100)
                plt.close()
                count += 1
                if count >= 4: break
            if count >= 4: break

    print(f"12 plots generated in {output_plot_dir}")

if __name__ == "__main__":
    generate_validation_plots()
