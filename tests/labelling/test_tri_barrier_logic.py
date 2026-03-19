import pytest
import polars as pl
import numpy as np
from pathlib import Path
from src.cloud.base_model.labelling.run_labelling import label_triple_barrier

def test_triple_barrier_logic_synthetic():
    """
    Testa a lógica da Tripla Barreira Refatorada (v8.2) com dados sintéticos.
    """
    # 1. Criar dados sintéticos (1 min de intervalo entre barras para teste fácil)
    # Ticks com 1 minuto real (60,000 ms)
    ts = [1000000 + i * 60000 for i in range(100)]
    
    # Preço simulado: sobe devagar, depois explode, depois cai.
    # Island_id constante (0)
    prices = [100.0] * 100
    for i in range(1, 10): prices[i] = prices[i-1] * 1.0001 # Flat
    prices[10] = 105.0 # Spike BUY (hits TP quickly)
    for i in range(11, 20): prices[i] = 105.0
    prices[20] = 90.0  # Spike SELL (hits SL quickly)
    
    df = pl.DataFrame({
        "ts": ts,
        "close": prices,
        "high": [p * 1.0001 for p in prices],
        "low": [p * 0.9999 for p in prices],
        "island_id": [0] * 100
    })
    
    config = {
        "pre_processing": {
            "labelling": {
                "horizon_minutes": 5,     # Window 5m
                "pt_multiplier": 2.0,
                "sl_multiplier": 2.0,
                "vol_span": 20
            }
        },
        "execution": {
            "taker_fee_pct": 0.000
        }
    }
    
    # Precisamos de um arquivo fake para satisfazer a assinatura da função
    dummy_file = Path("dummy.parquet")
    df.write_parquet(dummy_file)
    
    try:
        df_labelled = label_triple_barrier([dummy_file], config)
        
        # Verificações
        # 1. At row 0-4 (before spike at 10), we don't have enough volatility yet 
        # or the spike is outside the 5m window.
        # But at row 6, 7, 8, 9, the 5m window (10, 11, 12, 13, 14) WILL hit the price 105.0.
        
        labels = df_labelled["target"].to_list()
        
        # O spike em 105 ocorre no índice 10.
        # Janela de 5m (rows 10, 11, 12, 13, 14).
        # Para a row 6 (ts=1.36m), a janela 5m termina em ts=1.66m (row 11).
        # Então a row 6 deve ver a high=105 no futuro e marcar BUY (2).
        
        # Testar se temos classes BUY e SELL
        assert 2 in labels, "BUY label not generated"
        assert 0 in labels, "SELL label not generated"
        assert 1 in labels, "NEUTRAL label not generated"
        
    finally:
        if dummy_file.exists():
            dummy_file.unlink()

if __name__ == "__main__":
    test_triple_barrier_logic_synthetic()
