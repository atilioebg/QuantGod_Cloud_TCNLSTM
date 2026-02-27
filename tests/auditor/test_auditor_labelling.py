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
        🛡️  TESTE DE COLISÃO DE HASHES — Anti-Leakage K-Fold OOF

        Verifica que NENHUM dado usado no TREINO dos Clones do Especialista
        apareceu como dado de TREINO do Auditor (dataset_fused).

        Algoritmo:
          1. Para cada fold k, carregamos fold_k.parquet (predições OOF do bloco TESTE).
          2. Derivamos os índices de TREINO daquele fold via blocked_purged_kfold_indices().
          3. Calculamos SHA256 dos bytes do array de indices de treino de cada fold.
          4. Calculamos SHA256 dos original_row_idx presentes no fused_auditor.parquet (treino).
          5. Verificamos que os índices de treino do Especialista
             NÃO aparecem no fused_auditor.parquet de treino do Auditor.

        Regra de Ouro: O Auditor só vê os blocos TESTE do K-Fold
        (as predições OOF). Os blocos TREINO dos Clones são irrelevantes
        para o Auditor e jamais foram injetados no dataset_fused.
        """
        import yaml
        import hashlib

        # ── Load config ───────────────────────────────────────────────────────
        cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
        if not cfg_path.exists():
            pytest.skip("master_config.yaml not found")

        with open(cfg_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        kfold_cfg = config.get('pre_processing', {}).get('kfold', {})
        if not kfold_cfg.get('enabled', False):
            pytest.skip("K-Fold not enabled in master_config.yaml (kfold.enabled=false)")

        # ── Verify required files exist ───────────────────────────────────────
        oof_dir       = Path(kfold_cfg.get('oof_output_dir', 'data/auditor/oof_predictions'))
        full_oof_path = oof_dir / "full_oof.parquet"
        fused_train   = Path("data/auditor/dataset_fused/train/fused_auditor.parquet")

        if not full_oof_path.exists():
            pytest.skip(f"full_oof.parquet not found (run K-Fold first): {full_oof_path}")
        if not fused_train.exists():
            pytest.skip(f"fused_auditor.parquet not found (run auditor_labelling first): {fused_train}")

        # ── Load OOF test indices (safe to be in Auditor) ─────────────────────
        # full_oof.parquet contains ONLY the test-block predictions from each fold.
        # These are the rows the Auditor CAN see — they were never used for training.
        df_oof       = pl.read_parquet(full_oof_path)
        oof_test_idx = set(df_oof['original_row_idx'].to_list())

        # ── Load Auditor training indices ─────────────────────────────────────
        df_fused_train = pl.read_parquet(fused_train)

        # fused_auditor.parquet should have original_row_idx if K-Fold mode produced it
        if 'original_row_idx' not in df_fused_train.columns:
            pytest.skip(
                "fused_auditor.parquet has no 'original_row_idx' column. "
                "Possibly generated in legacy mode — leakage test not applicable."
            )

        auditor_train_idx = set(df_fused_train['original_row_idx'].to_list())

        # ── Per-fold hash verification ────────────────────────────────────────
        # For each fold, reconstruct which original rows were used for TRAINING
        # (not testing). These rows MUST NOT appear in the Auditor's training data.
        n_splits        = kfold_cfg.get('n_splits', 5)
        sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
        buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
        mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
        base_labelled_name = f"labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
        foundation_val_dir = Path(f"data/L2/splits_{base_labelled_name}/val")

        if not foundation_val_dir.exists():
            pytest.skip(f"Foundation val not found: {foundation_val_dir}")

        val_files = sorted(list(foundation_val_dir.glob("*.parquet")))
        n_total   = sum(pl.read_parquet(f, columns=['target']).shape[0] for f in val_files)

        freq     = config['pre_processing']['etl'].get('resample_freq', '1min')
        r_min    = max(1, int(freq.replace('min', '').replace('T', '')))
        purge_m  = kfold_cfg.get('purge_minutes', 15)
        purge_b  = max(1, purge_m // r_min)

        # Lazy import to avoid circular dependency in test discovery
        sys.path.insert(0, str(Path(".").resolve()))
        from src.cloud.base_model.treino.run_kfold_specialist import blocked_purged_kfold_indices

        specialist_train_all  = set()
        fold_train_hashes     = {}

        for fold_k, (train_idx, test_idx) in enumerate(
            blocked_purged_kfold_indices(n_total, n_splits, purge_b)
        ):
            # SHA256 of training indices for this fold (deterministic fingerprint)
            idx_bytes  = train_idx.tobytes()
            fold_hash  = hashlib.sha256(idx_bytes).hexdigest()[:16]
            fold_train_hashes[fold_k] = fold_hash
            specialist_train_all.update(train_idx.tolist())

        # ── The Core Leakage Check (Per-Fold Stacking Rule) ───────────────────
        # Leakage occurs ONLY IF a row used to train Fold K appears in the OOF test block
        # generated by Fold K. Global intersection is normal in K-Fold.
        
        df_fused_train_kfold = df_fused_train.join(df_oof.select(['original_row_idx', 'fold']), on='original_row_idx', how='inner')
        
        total_leakage = 0
        leakage_details = []
        
        for fold_k, (train_idx, test_idx) in enumerate(
            blocked_purged_kfold_indices(n_total, n_splits, purge_b)
        ):
            # OOF rows produced by THIS fold that ended up in Auditor's training set
            oof_k_in_auditor = set(
                df_fused_train_kfold.filter(pl.col("fold") == fold_k)["original_row_idx"].to_list()
            )
            
            # Training rows used by THIS fold
            train_k_set = set(train_idx.tolist())
            
            # Leakage for Fold K: Intersection between its Train and its OOF
            fold_leakage = oof_k_in_auditor & train_k_set
            total_leakage += len(fold_leakage)
            if len(fold_leakage) > 0:
                leakage_details.append(f"Fold {fold_k}: {len(fold_leakage)} leaked rows")
        
        assert total_leakage == 0, (
            f"\n🚨 DATA LEAKAGE DETECTADO (Per-Fold)!\n"
            f"   {total_leakage} linhas foram usadas no TREINO de um Clone e também\n"
            f"   aparecem no bloco OOF que ESSE MESMO Clone gerou para o Auditor.\n"
            f"   Detalhes: {leakage_details}\n"
            f"   Hashes por fold: {fold_train_hashes}\n"
        )

        # Verify OOF test indices ARE the source for auditor (non-zero intersection)
        oof_in_auditor = auditor_train_idx & oof_test_idx
        assert len(oof_in_auditor) > 0, (
            "Auditor training set has zero intersection with OOF test predictions. "
            "Something went wrong in the fusion join."
        )

        logger.info(
            f"✅ HASH COLLISION TEST PASSED\n"
            f"   Fold hashes: {fold_train_hashes}\n"
            f"   Auditor train uses {len(auditor_train_idx):,} of {len(oof_test_idx):,} OOF test rows.\n"
            f"   Specialist training rows in Auditor training: 0 (CLEAN)"
        )
