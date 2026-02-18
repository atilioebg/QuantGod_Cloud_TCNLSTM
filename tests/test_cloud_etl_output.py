import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Configuração do caminho dos dados de teste
TEST_DATA_DIR = Path("data/L2/pre_processed")

def get_test_files():
    """Retorna a lista de arquivos parquet na pasta de teste."""
    return list(TEST_DATA_DIR.glob("*.parquet"))

@pytest.mark.parametrize("file_path", get_test_files())
class TestCloudDataIntegrity:
    
    def test_column_count(self, file_path):
        """Valida se o arquivo possui as 810 colunas (200 níveis + 9 features + close)."""
        df = pd.read_parquet(file_path)
        # 200 bids (p,s) + 200 asks (p,s) = 800
        # + 9 features agregadas + 1 close = 10
        # Total = 810
        assert len(df.columns) == 810, f"O arquivo {file_path.name} deveria ter 810 colunas, mas tem {len(df.columns)}"

    def test_essential_columns(self, file_path):
        """Valida se as 9 features de treinamento e a coluna close existem."""
        df = pd.read_parquet(file_path)
        essential_cols = [
            'body', 'upper_wick', 'lower_wick', 'log_ret_close',
            'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume',
            'close'
        ]
        for col in essential_cols:
            assert col in df.columns, f"Coluna essencial '{col}' ausente em {file_path.name}"

    def test_orderbook_sorting(self, file_path):
        """Valida a ordenação de preços do Orderbook (Bids desc, Asks asc)."""
        df = pd.read_parquet(file_path)
        # Pegamos uma amostra do meio do arquivo para performance
        sample = df.iloc[len(df)//2]
        
        # Bids devem ser decrescentes
        bids_p = [sample[f'bid_{i}_p'] for i in range(200)]
        for i in range(len(bids_p)-1):
            if not np.isnan(bids_p[i+1]):
                assert bids_p[i] >= bids_p[i+1], f"Erro de ordenação nos Bids no nível {i} em {file_path.name}"
                
        # Asks devem ser crescentes
        asks_p = [sample[f'ask_{i}_p'] for i in range(200)]
        for i in range(len(asks_p)-1):
            if not np.isnan(asks_p[i+1]):
                assert asks_p[i] <= asks_p[i+1], f"Erro de ordenação nos Asks no nível {i} em {file_path.name}"

    def test_no_book_crossing(self, file_path):
        """Valida que o melhor Bid é sempre menor que o melhor Ask (Spread positivo)."""
        df = pd.read_parquet(file_path)
        # O spread deve ser positivo em todas as linhas
        assert (df['bid_0_p'] < df['ask_0_p']).all(), f"Detectado book cruzado (Bid >= Ask) em {file_path.name}"

    def test_data_quality_no_nans_in_features(self, file_path):
        """Garante que as 9 features de treino não possuem NaNs."""
        df = pd.read_parquet(file_path)
        feature_cols = [
            'body', 'upper_wick', 'lower_wick', 'log_ret_close',
            'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
        ]
        nan_counts = df[feature_cols].isna().sum().sum()
        assert nan_counts == 0, f"Detectados {nan_counts} NaNs nas colunas de features em {file_path.name}"

    def test_chronological_order(self, file_path):
        """Valida se os dados estão em ordem crescente."""
        df = pd.read_parquet(file_path)
        # Se 'ts' estiver presente, usamos como referência
        if 'ts' in df.columns:
            assert (df['ts'].diff().dropna() >= 0).all(), f"O arquivo {file_path.name} não está em ordem cronológica"
        else:
            assert df.index.is_monotonic_increasing, f"O arquivo {file_path.name} não está em ordem cronológica"
