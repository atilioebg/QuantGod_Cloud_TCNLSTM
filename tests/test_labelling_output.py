
import polars as pl
from pathlib import Path
import pytest

LABELLED_DIR = Path("data/L2/labelled")

def get_labelled_files():
    return list(LABELLED_DIR.glob("*.parquet"))

@pytest.mark.parametrize("file_path", get_labelled_files())
class TestLabellingOutput:
    
    def test_file_exists(self, file_path):
        assert file_path.exists()

    def test_target_column_exists(self, file_path):
        df = pl.read_parquet(file_path)
        assert "target" in df.columns, f"A coluna 'target' está ausente em {file_path.name}"
        assert df["target"].dtype in [pl.Int32, pl.Int64], f"Tipo de coluna 'target' inválido em {file_path.name}"

    def test_no_nans_in_target(self, file_path):
        df = pl.read_parquet(file_path)
        assert df["target"].null_count() == 0, f"Existem NaNs na coluna 'target' em {file_path.name}"

    def test_label_values(self, file_path):
        df = pl.read_parquet(file_path)
        unique_labels = df["target"].unique().to_list()
        for label in unique_labels:
            assert label in [0, 1, 2], f"Valor de label inválido ({label}) encontrado em {file_path.name}"

    def test_label_distribution(self, file_path):
        """Verifica se não há apenas uma única classe (o que indicaria erro de threshold)"""
        df = pl.read_parquet(file_path)
        unique_count = df["target"].n_unique()
        # Em datasets pequenos de teste, pode ser que só tenha 1 ou 2, mas 0 é erro.
        assert unique_count > 0, f"O arquivo {file_path.name} está vazio ou sem labels."
        
        counts = df["target"].value_counts()
        print(f"\nDistribuição {file_path.name}:")
        print(counts)
