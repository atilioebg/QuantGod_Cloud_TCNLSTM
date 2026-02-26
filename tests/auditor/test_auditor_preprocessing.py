import pytest
import pandas as pd
import polars as pl
from pathlib import Path
import os
import numpy as np

# A suite deve rodar com base nos arquivos locais
CONTEXT_DIR = Path("data/auditor/context/train")

def get_context_files():
    if not CONTEXT_DIR.exists():
        return []
    return sorted(list(CONTEXT_DIR.glob("*.parquet")))

class TestAuditorPreprocessing:
    def test_directory_has_parquet_files(self):
        """Verifica se a engenharia de contexto salvou os dados."""
        assert CONTEXT_DIR.exists(), f"Diretório de contexto não encontrado: {CONTEXT_DIR}"
        files = get_context_files()
        assert len(files) > 0, "Nenhum arquivo .parquet de contexto encontrado na pasta."

    @pytest.mark.parametrize("file_path", get_context_files())
    def test_column_structure_and_count(self, file_path):
        """Valida se as novas colunas Alpha (ADX, VWAP, MFI, Skewness) estão injetadas."""
        df = pl.read_parquet(file_path).to_pandas()
        
        required_features = [
            'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
            'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
            # v4.2 Alpha sensors
            'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
        ]
        
        for feat in required_features:
            assert feat in df.columns, f"A feature essencial '{feat}' está ausente em {file_path.name}"
            
    @pytest.mark.parametrize("file_path", get_context_files())
    def test_no_nans_in_context(self, file_path):
        """Garante que a matemática aplicada nas features de contexto L2 não vazaram NaNs ou Infs."""
        df = pl.read_parquet(file_path).to_pandas()
        
        columns_to_check = [col for col in df.columns if col not in ['target', 'ts']]
        
        nan_counts = df[columns_to_check].isna().sum().sum()
        assert nan_counts == 0, f"Detectados {nan_counts} NaNs nas colunas de features/context em {file_path.name}"
        
        inf_counts = np.isinf(df[columns_to_check].values).sum()
        assert inf_counts == 0, f"Detectados {inf_counts} valores Infinitos nas features em {file_path.name}"
