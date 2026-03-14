import pandas as pd
import numpy as np
from pathlib import Path

def analyze_performance():
    file_path = Path("logs/trading_journal.csv")
    if not file_path.exists():
        print("❌ Arquivo logs/trading_journal.csv não encontrado.")
        return

    df = pd.read_csv(file_path)
    
    # Filtrar apenas sinais de BUY e SELL (Neutro não tem 'was_correct' definido da mesma forma)
    df_trades = df[df['prediction'].isin(['BUY', 'SELL'])].copy()
    
    if len(df_trades) == 0:
        print("⚠️ Nenhuma operação de BUY/SELL registrada no histórico ainda.")
        return

    # Converter para numérico caso necessário
    df_trades['was_correct'] = df_trades['was_correct'].astype(bool)
    
    print("="*60)
    print("📊 RELATÓRIO DE PERFORMANCE (QuantGod v5.5)")
    print("="*60)
    
    # Estatísticas Gerais
    total = len(df_trades)
    hits = df_trades['was_correct'].sum()
    win_rate = (hits / total) * 100
    avg_pnl = df_trades['pnl_observed_pct'].mean()
    max_pnl = df_trades['pnl_observed_pct'].max()
    
    print(f"Total de Previsões (BUY/SELL): {total}")
    print(f"Taxa de Acerto (Hit Rate):      {win_rate:.2f}%")
    print(f"PnL Médio (Potencial):         {avg_pnl:.2f}%")
    print(f"Maior PnL na Janela:           {max_pnl:.2f}%")
    
    # Análise por Faixa de Confiança
    print("\n🔍 ANÁLISE POR CONFIANÇA (AUDITOR SCORE):")
    bins = [0.68, 0.75, 0.82, 0.90, 1.0]
    labels = ['68-75% (Baixa)', '75-82% (Média)', '82-90% (Alta)', '90-100% (Crítica)']
    
    df_trades['confidence_range'] = pd.cut(df_trades['auditor_score'], bins=bins, labels=labels)
    
    perf_by_conf = df_trades.groupby('confidence_range', observed=False).agg({
        'was_correct': ['count', 'sum', 'mean'],
        'pnl_observed_pct': 'mean'
    })
    
    # Renomear colunas para facilitar leitura
    perf_by_conf.columns = ['Total', 'Hits', 'Win Rate', 'Avg PnL']
    perf_by_conf['Win Rate'] = perf_by_conf['Win Rate'] * 100
    
    print(perf_by_conf.to_string())
    
    # Análise de "Quase Acertos"
    # Consideramos quase acerto se PnL > 0.15% (metade do alvo)
    near_misses = df_trades[(df_trades['was_correct'] == False) & (df_trades['pnl_observed_pct'] >= 0.15)].shape[0]
    print(f"\n🎯 Quase Acertos (>0.15%): {near_misses} ({ (near_misses/total)*100:.1f}% das falhas)")
    
    print("="*60)

if __name__ == "__main__":
    analyze_performance()
