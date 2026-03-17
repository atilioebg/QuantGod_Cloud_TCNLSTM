import pytest
import polars as pl
import numpy as np
from src.cloud.base_model.pre_processamento.etl.event_sampler import EventSampler

@pytest.fixture
def dummy_data():
    """Generates synthetic trades data strictly ordered with simple gaps."""
    # 5 ticks total
    return pl.DataFrame({
        "ts": [
            1000000,           # island 0
            1001000, 
            1005000,
            # gap 10 min
            1605000,           # island 1
            1606000
        ],
        "usd_volume": [20000, 50000, 40000, 110000, 10000],
        "size": [0.2, 0.5, 0.4, 1.1, 0.1],
        "price": [100000, 100000, 100000, 100000, 100000],
        "spread": [0.1, 0.1, 0.1, 0.1, 0.1],
        "micro_price": [100000, 100000, 100000, 100000, 100000]
    })

@pytest.fixture
def mock_cfg():
    return {
        "bar_type": "dollar",
        "dollar_threshold_usd": 100000.0,
        "island_gap_minutes": 5.0,
        "resample_min": 5,
        "delta_short_min": 5,
        "delta_long_min": 30,
        "vpin_window_min": 25,
        "spread_zscore_window_min": 60,
    }

class TestEventSampler:
    def test_volume_reconciliation(self, dummy_data, mock_cfg):
        """Volume Check: Integridade de Massa. Sum(Bar_Volume) == Sum(Raw_Trades)"""
        sampler = EventSampler(mock_cfg)
        bars = sampler.compute_event_bars(dummy_data)
        
        sum_raw = dummy_data["usd_volume"].sum()
        sum_bars = bars["bar_usd_volume"].sum()
        
        # Erro < 0.0001%
        assert np.isclose(sum_raw, sum_bars, rtol=1e-6), f"Volume mismatch: Raw={sum_raw}, Bars={sum_bars}"

    def test_transbordo_dollar_invariance(self, dummy_data, mock_cfg):
        """Dollar Invariance/Tick de Transbordo. A primeira barra deve conter 20k+50k+40k = 110k"""
        sampler = EventSampler(mock_cfg)
        bars = sampler.compute_event_bars(dummy_data)
        
        vols = bars["bar_usd_volume"].to_list()
        
        # O tick acumulado na ilha 0
        # Tick0=20k -> cum 20k -> shift 0 -> id 0
        # Tick1=50k -> cum 70k -> shift 20k -> id 0
        # Tick2=40k -> cum 110k -> shift 70k -> id 0 (overflows next!)
        # Na ilha 1 (reseta)
        # Tick3=110k -> cum 110k -> shift 0 -> id 0
        # Tick4=10k -> cum 120k -> shift 110k -> id 1
        
        assert len(vols) == 3 
        assert vols[0] == 110000.0  # (id 0, ilha 0)
        assert vols[1] == 110000.0  # (id 0, ilha 1)
        assert vols[2] == 10000.0   # (id 1, ilha 1)

    def test_leakage_audit_timestamp(self, dummy_data, mock_cfg):
        """Leakage Audit: Garantir que a barra seja carimbada EXATAMENTE com o ms do último trade (transbordo)"""
        sampler = EventSampler(mock_cfg)
        bars = sampler.compute_event_bars(dummy_data)
        
        ts_list = bars["ts"].to_list()
        # A barra 0 da ilha 0 fecha no Tick2 (ts = 1005000)
        assert ts_list[0] == 1005000
