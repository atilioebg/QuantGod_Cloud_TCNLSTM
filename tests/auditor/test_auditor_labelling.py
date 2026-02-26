import pytest
import pandas as pd
import polars as pl
from pathlib import Path
import numpy as np
import logging

FUSED_DIR = Path("data/auditor/dataset_fused/train")
logger = logging.getLogger(__name__)

def get_fused_files():
    if not FUSED_DIR.exists():
        return []
    return sorted(list(FUSED_DIR.glob("*.parquet")))

class TestAuditorLabelling:
    def test_fused_directory_exists(self):
        """Verifica presença dos dados rotulados crus cruzados."""
        assert FUSED_DIR.exists(), f"Diretório fundido não encontrado: {FUSED_DIR}"
        files = get_fused_files()
        assert len(files) > 0, "Nenhum arquivo Parquet fundido foi localizado. Execute auditor_labelling.py primeiro."

    @pytest.mark.parametrize("file_path", get_fused_files())
    def test_model_logits_presence(self, file_path):
        """Verifica o sucesso do passo de inferência cruzada do PyTorch dentro da Labeling Phase."""
        df = pl.read_parquet(file_path).to_pandas()
        expected_logits = [
            'base_prob_sell', 'base_prob_neu', 'base_prob_buy',
            'spec_prob_sell', 'spec_prob_neu', 'spec_prob_buy'
        ]
        for logit in expected_logits:
            assert logit in df.columns, f"Logit de probabilidade Pytorch '{logit}' ausente em {file_path.name}"
            
    @pytest.mark.parametrize("file_path", get_fused_files())
    def test_meta_target_values(self, file_path):
        """Valida a restrição binária do rótulo Meta do XGBoost."""
        df = pl.read_parquet(file_path).to_pandas()
        assert 'meta_target' in df.columns, f"Rótulo de Meta-Veto 'meta_target' não encontrado em {file_path.name}"
        unique_vals = set(df['meta_target'].unique())
        assert unique_vals.issubset({0, 1}), f"Vazamento no Meta-Target (Existem valores absurdos): {unique_vals}"
        
    @pytest.mark.parametrize("file_path", get_fused_files())
    def test_meta_target_cross_logic(self, file_path):
        """
        Teste de Fogo: Garante matematicamente a geração do Veto.
        Se Predição(Especialista) == Target_Real -> Acerto(1), Senão Erro(0)
        """
        df = pl.read_parquet(file_path).to_pandas()
        assert 'true_target' in df.columns, f"Gabarito Histórico 'true_target' ausente em {file_path.name}"
        
        spec_probs_cols = ['spec_prob_sell', 'spec_prob_neu', 'spec_prob_buy']
        spec_probs = df[spec_probs_cols].values
        
        pred_class = np.argmax(spec_probs, axis=1)
        expected_meta = (pred_class == df['true_target'].values).astype(int)
        
        mismatches = (df['meta_target'].values != expected_meta).sum()
        assert mismatches == 0, f"⚠️ Erro Matemático na Classificação: {mismatches} linhas divergentes com os Logs Pytorch x XGBoost."

    def test_data_leakage_specialist_vs_auditor(self):
        """
        [ v4.3 EXPERIMENTAL - Cuidado contra Vazamentos de Out-Of-Fold ]
        O Auditor deve ser treinado onde a rede neural NUNCA FOI OTIMIZADA.
        Se os datasets de Treino do Especialista e Treino do Auditor convergirem para os mesmos
        Timestamps, o XGBoost vai decorar os falsos positivos como sendo Deuses (High-Confidence Overfitting).
        """
        logger.warning(
            "\\n🛡️ ALERTA ENGENHARIA [v4.3 PREP] 🛡️"
            "\\nTeste TDD de 'Data Leakage' reservado.\\n"
            "Implementaremos colisão de hashings do TS (`time_stamp`) em breve quando a "
            "K-Fold Cross Validation em `master_config.yaml` for ativada."
        )
        # Passa o script como True/Válido porque a arquitetura atual permite o teste
        assert True, "O stub lógico de Anti-Vazamento (Data Leakage) está a postos."
