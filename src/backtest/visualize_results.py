import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_backtest():
    # Caminho do arquivo gerado
    file_path = Path("data/backtest/L2/backtest_results_COMPLETE.parquet")
    if not file_path.exists():
        print(f"❌ Arquivo não encontrado em {file_path}")
        return

    print("📖 Carregando dados...")
    df = pd.read_parquet(file_path)
    
    # Converter ts para datetime se necessário
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.sort_values('ts')
    
    # Calcular PnL acumulado em base $1000
    initial_capital = 1000
    df['equity'] = initial_capital + (initial_capital * df['pnl_net'].cumsum())
    
    print("📈 Gerando gráfico...")
    plt.figure(figsize=(15, 8))
    plt.plot(df['ts'], df['equity'], label='Equity Curve (Net PnL)', color='#00ff88', linewidth=1.5)
    
    plt.title('QuantGod High-Frequency Backtest: TCN-LSTM + Auditor Ensemble', fontsize=16, color='white')
    plt.xlabel('Data/Hora', fontsize=12, color='white')
    plt.ylabel('Capital (USD)', fontsize=12, color='white')
    
    # Estilização Dark Mode Premium
    plt.gcf().set_facecolor('#0a0a0a')
    plt.gca().set_facecolor('#0a0a0a')
    plt.gca().tick_params(colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.grid(True, alpha=0.1, color='white')
    plt.legend()
    
    output_img = "data/backtest/L2/equity_curve.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo com sucesso em: {output_img}")
    
    # Resumo final
    print("\n--- RESUMO DE RENTABILIDADE ---")
    print(f"Capital Inicial: ${initial_capital:.2f}")
    print(f"Capital Final:   ${df['equity'].iloc[-1]:.2f}")
    print(f"Lucro Líquido:   ${df['equity'].iloc[-1] - initial_capital:.2f}")
    print(f"Retorno Total:   {df['pnl_net'].sum()*100:.2f}%")
    print("-------------------------------")

if __name__ == "__main__":
    plot_backtest()
