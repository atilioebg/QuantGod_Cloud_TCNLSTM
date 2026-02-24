import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path

# =============================================================================
# CONFIGURAÇÃO DE DIRETÓRIO (Compatível com RunPod e Local)
# =============================================================================
def get_default_dir():
    # Agora usamos apenas paths relativos, assumindo que o teste roda da raiz do repo
    rel_path = Path("data/L2/pre_processed")
    return rel_path

TEST_DATA_DIR = Path(os.getenv("PRE_PROCESSED_DIR", get_default_dir()))

def get_test_files():
    """Retorna a lista de arquivos parquet na pasta de teste."""
    files = sorted(list(TEST_DATA_DIR.glob("*.parquet")))
    return files

# =============================================================================
# DEFINIÇÃO DE CONSTANTES DO DATASET (QuantGod v10 - 16 Features)
# =============================================================================
DYNAMIC_FEATURES = [
    'ofi', 'ofi_delta_5', 'ofi_delta_1',
    'micro_price_momentum', 'micro_price_delta_5', 'micro_price_delta_1',
    'bid_slope', 'ask_slope', 
    'bid_rdi', 'bid_rdi_delta_5', 'bid_rdi_delta_1',
    'ask_rdi', 'ask_rdi_delta_5', 'ask_rdi_delta_1',
    'book_asymmetry_v5', 'spread_zscore_60', 'vpin_lite_5', 'pressure_ratio'
]
AGG_FEATURES = [
    'body', 'upper_wick', 'lower_wick', 'log_ret_close',
    'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
]
ALL_FEATURES = AGG_FEATURES + DYNAMIC_FEATURES

# OB200: 200 bids (p,s) + 200 asks (p,s) = 800 colunas
OB_LEVELS = 200
OB_COLS = []
for i in range(OB_LEVELS):
    OB_COLS.extend([f"bid_{i}_p", f"bid_{i}_s", f"ask_{i}_p", f"ask_{i}_s"])

# Total esperado: 27 features + 1 close + 800 OB = 828 colunas
EXPECTED_COL_COUNT = len(ALL_FEATURES) + 1 + len(OB_COLS)

# =============================================================================
# SUITE DE TESTES
# =============================================================================

@pytest.mark.parametrize("file_path", get_test_files())
class TestCloudDataIntegrity:
    
    def test_file_exists_and_readable(self, file_path):
        """Valida se o arquivo existe e pode ser lido pelo pandas/pyarrow."""
        assert file_path.exists(), f"Arquivo não encontrado: {file_path}"
        df = pd.read_parquet(file_path)
        assert not df.empty, f"Arquivo {file_path.name} está vazio."

    def test_column_structure_and_count(self, file_path):
        """Valida se o arquivo possui as 817 colunas e se todas as features estão presentes."""
        df = pd.read_parquet(file_path)
        
        # 1. Check Count
        assert len(df.columns) == EXPECTED_COL_COUNT, \
            f"Erro em {file_path.name}: Esperado {EXPECTED_COL_COUNT} colunas, encontrado {len(df.columns)}"
        
        # 2. Check Feature Names
        for feat in ALL_FEATURES + ['close']:
            assert feat in df.columns, f"Coluna essencial '{feat}' ausente em {file_path.name}"
            
        # 3. Check Orderbook Columns
        # Validamos apenas o primeiro e o último nível para evitar overhead excessivo
        for col in ["bid_0_p", "bid_199_p", "ask_0_p", "ask_199_p"]:
            assert col in df.columns, f"Coluna de orderbook '{col}' ausente em {file_path.name}"

    def test_data_quality_no_nans_or_infs(self, file_path):
        """Garante que as 16 features de treino não possuem NaNs ou Valores Infinitos."""
        df = pd.read_parquet(file_path)
        
        # 1. Check NaNs
        nan_counts = df[ALL_FEATURES].isna().sum().sum()
        assert nan_counts == 0, f"Detectados {nan_counts} NaNs nas colunas de features em {file_path.name}"
        
        # 2. Check Infs
        inf_counts = np.isinf(df[ALL_FEATURES].values).sum()
        assert inf_counts == 0, f"Detectados {inf_counts} valores Infinitos nas features em {file_path.name}"

    def test_orderbook_integrity(self, file_path):
        """Valida a saúde do Orderbook (Sorting, No Crossing, Positive Prices)."""
        df = pd.read_parquet(file_path)
        
        # 1. No Book Crossing (Best Bid < Best Ask)
        # Permite NaN se o book estiver vazio, mas se houver preço, o spread deve ser > 0
        mask = df['bid_0_p'].notna() & df['ask_0_p'].notna()
        assert (df.loc[mask, 'bid_0_p'] < df.loc[mask, 'ask_0_p']).all(), \
            f"Detectado book cruzado (Bid >= Ask) em {file_path.name}"
            
        # 2. Positive Prices
        assert (df['close'] > 0).all(), f"Preço de fechamento negativo ou zero em {file_path.name}"
        
        # 3. Sorting (Amostra randômica de 10 linhas para performance)
        sample_indices = np.random.choice(df.index, min(len(df), 10), replace=False)
        for idx in sample_indices:
            row = df.loc[idx]
            # Bids: decrescente
            bids = [row[f'bid_{i}_p'] for i in range(10) if not np.isnan(row[f'bid_{i}_p'])]
            assert all(bids[i] >= bids[i+1] for i in range(len(bids)-1)), f"Bids desordenados em {file_path.name} na linha {idx}"
            # Asks: crescente
            asks = [row[f'ask_{i}_p'] for i in range(10) if not np.isnan(row[f'ask_{i}_p'])]
            assert all(asks[i] <= asks[i+1] for i in range(len(asks)-1)), f"Asks desordenados em {file_path.name} na linha {idx}"

    def test_chronological_order(self, file_path):
        """Valida se os dados estão em ordem crescente de tempo."""
        df = pd.read_parquet(file_path)
        # Se 'ts' estiver configurado como índice ou coluna
        if 'ts' in df.columns:
            assert df['ts'].is_monotonic_increasing, f"Arquivo {file_path.name} não está em ordem cronológica (coluna ts)"
        else:
            assert df.index.is_monotonic_increasing, f"Arquivo {file_path.name} não está em ordem cronológica (índice)"

    def test_data_types(self, file_path):
        """Garante que as colunas numéricas não foram corrompidas para objetos/strings."""
        df = pd.read_parquet(file_path)
        # Todas as features devem ser float ou int
        for feat in ALL_FEATURES:
            assert pd.api.types.is_numeric_dtype(df[feat]), f"Coluna {feat} em {file_path.name} não é numérica"
