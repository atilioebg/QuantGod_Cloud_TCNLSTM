"""
auditor_labelling.py — Geração do Dataset Fundido para o Auditor XGBoost

Modo de operação detectado automaticamente via master_config.yaml:

  kfold.enabled = true  →  MODO K-FOLD OOF (Fase 2 Engorda Total)
    Fonte de predições: data/auditor/oof_predictions/full_oof.parquet
    Alinhamento: inner join por `original_row_idx` com context features.
    Resultado: ~100% do Foundation Val disponível para o Auditor (~60k–300k rows).

  kfold.enabled = false →  MODO LEGADO (Holdout Especialista)
    Fonte de predições: inferência direta do modelo Especialista no spec_val_dir.
    Alinhamento: offset posicional por seq_len.
    Resultado: ~20% do Foundation Val (~10k rows).

Output (ambos os modos):
  data/auditor/dataset_fused/train/fused_auditor.parquet
  data/auditor/dataset_fused/val/fused_auditor.parquet
"""

import torch
from torch.utils.data import DataLoader, Dataset
import polars as pl
import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
import sys
import pickle
import json

project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.models.model import Hybrid_TCN_LSTM
from src.cloud.base_model.utils.logging_utils import setup_logger
from src.cloud.base_model.utils.path_utils import get_labelled_dir, get_auditor_context_dir, get_drive_session_path, resolve_local_drive

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Island-aware sliding window dataset.
    
    Janelas que cruzam fronteiras de ilhas são descartadas para evitar que
    o modelo aprenda padrões artificiais criados por gaps de mercado.
    Idêntico à versão em run_kfold_specialist.py.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, island_ids: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

        max_idx = len(X) - seq_len
        if max_idx < 0:
            self.valid_indices = np.array([], dtype=np.int64)
        else:
            # island_ids[idx] == island_ids[idx + seq_len - 1] → sem cruzamento de ilha
            self.valid_indices = np.where(
                island_ids[:max_idx + 1] == island_ids[seq_len - 1:]
            )[0].astype(np.int64)

        logger.info(
            f"SequenceDataset (island-aware): {len(self.valid_indices)}/{max(0, max_idx + 1)} "
            f"sequências válidas (Lookback protection ativo)."
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x_seq   = self.X[i: i + self.seq_len]
        y_label = self.y[i + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.long)


def load_config():
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
            probs = torch.softmax(outputs["logits"], dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    return np.vstack(all_probs), np.concatenate(all_targets)


def _build_fused_and_save(df_fused: pd.DataFrame, config: dict, output_dir: str):
    """Common post-processing: meta-target, split and save to parquet."""
    # ── Meta-Labeling Logic ───────────────────────────────────────────────────
    # meta_target = 1 if Specialist prediction matches real target, else 0.
    spec_probs_cols = ['spec_prob_sell', 'spec_prob_neu', 'spec_prob_buy']
    spec_preds_class = np.argmax(df_fused[spec_probs_cols].values, axis=1)
    df_fused["meta_target"] = np.where(spec_preds_class == df_fused["true_target"].values, 1, 0)

    logger.info(
        f"📊 Fused Dataset: {len(df_fused):,} rows | "
        f"Meta-Target Accuracy: {df_fused['meta_target'].mean():.2%}"
    )

    # ── Chronological Split ───────────────────────────────────────────────────
    # Correção Patch 1: O Auditor DEVE treinar APENAS sobre as predições Out-of-Fold puras
    # que NÃO tenham sido usadas no treino de *nenhum* clone num setting não-kfold, ou no caso do K-Fold,
    # aceitar que o OOF é um construto de validação.
    # O teste de auditoria v4.4 avaliava que o Auditor Train não poderia conter NENHUM dado do Specialist Train.
    # Em um Stacking K-Fold, o OOF inteiro é usado pelo meta-modelo. O split aqui separa OOF-Train e OOF-Val
    # para o próprio early-stopping do Auditor.
    
    split_pct = config['pre_processing']['labelling']['split']['auditor']['train_ratio']
    split_idx = int(len(df_fused) * split_pct)

    train_fused = df_fused.iloc[:split_idx]
    val_fused   = df_fused.iloc[split_idx:]

    out_path = Path(output_dir)
    train_path = out_path / "train"
    val_path   = out_path / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(train_fused).write_parquet(train_path / "fused_auditor.parquet")
    pl.DataFrame(val_fused).write_parquet(val_path / "fused_auditor.parquet")

    logger.info(f"📦 Treino Fused salvo: {train_path / 'fused_auditor.parquet'} ({len(train_fused):,} rows)")
    logger.info(f"📦 Val   Fused salvo: {val_path / 'fused_auditor.parquet'} ({len(val_fused):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# MODO K-FOLD OOF (primary path — Engorda Total)
# ══════════════════════════════════════════════════════════════════════════════
def load_and_fuse_kfold(config: dict, context_dir: str, output_dir: str):
    """
    K-Fold Mode: loads full_oof.parquet (Specialist OOF predictions) and
    joins with context features (ADX, Skewness, VWAP) via original_row_idx.

    Also runs the Foundation model inference on the SAME Foundation Val rows
    so the Auditor has BOTH base and specialist signal for its meta-labeling.

    The join is implemented as an INNER JOIN to guarantee timestamp alignment —
    only rows present in BOTH the OOF predictions AND the context features
    are included in the final dataset. No positional-offset assumptions.
    """
    kfold_cfg = config['pre_processing']['kfold']
    
    # Unificacao de Path: O OOF agora vive na pasta do Especialista (MODELOS/SPECIALIST)
    base_model_dir = get_drive_session_path("MODELOS", config)
    oof_dir = resolve_local_drive(Path(base_model_dir) / kfold_cfg.get('oof_output_dir', 'SPECIALIST'))
    full_oof_path = oof_dir / "full_oof.parquet"

    if not full_oof_path.exists():
        logger.error(f"❌ full_oof.parquet NAO ENCONTRADO em: {full_oof_path}")
        logger.error("   ↳ Certifique-se que o Especialista K-Fold rodou e gerou o arquivo.")
        sys.exit(1)

    logger.info(f"🔗 K-Fold Mode: loading OOF predictions from {full_oof_path}")
    df_oof = pl.read_parquet(full_oof_path)
    logger.info(f"  ↳ OOF rows: {len(df_oof):,} (covering {df_oof['fold'].n_unique()} folds)")

    # ── Load Context Features (LAZY) ────────────────────
    # v10.15: O Auditor agora só carregará os arquivos de Contexto que pertencem aos recortes da Validação Foundation,
    # atribuindo dinamicamente o offset do "original_row_idx". Isso impede que o Polars gere OOM (RAM) escalando
    # o índice original varrendo os 274 arquivos inteiros de 89 milhões de linhas antes do Inner Join.
    # ── Foundation Model Inference on Foundation Val ──────────────────────────
    # The OOF covers Foundation Val rows. We need Foundation model probs too
    # for the Auditor to have both signals. We run inference on the same rows.
    feature_cols = config['model']['feature_names']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔑 [Audit Fix] DEVICE para inferência Foundation: {DEVICE}")

    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    foundation_val_dir = Path(get_labelled_dir(config)) / "val"
    
    # ── Foundation Model Selection & Parameter Loading ─────────────────────────
    paths = config.get('pipeline_paths', {})
    best_base_macro = resolve_local_drive(Path(base_model_dir) / paths.get('best_tcn_lstm_model', 'BASE_MODEL/best_tcn_lstm.pt'))
    best_base_dir   = resolve_local_drive(Path(base_model_dir) / paths.get('best_tcn_lstm_dir_model', 'BASE_MODEL/best_tcn_lstm_dir.pt'))
    
    use_dir_strategy = config.get('training', {}).get('specialization_weights', {}).get('use_best_f1_dir', True)
    primary_base = best_base_dir if use_dir_strategy else best_base_macro
    fallback_base = best_base_macro if use_dir_strategy else best_base_dir
    active_base_path = primary_base if primary_base.exists() else (fallback_base if fallback_base.exists() else None)

    if not active_base_path:
        logger.error(f"[FAIL] MODELO BASE AUSENTE PARA FUSION: {primary_base}")
        sys.exit(1)

    # Choice of parameters JSON must match the chosen weights file
    is_dir_model = (active_base_path == best_base_dir)
    param_file = "best_dir_params.json" if is_dir_model else "best_params.json"
    base_params_path = Path("src/cloud/base_model/otimizacao") / param_file
    
    if not base_params_path.exists():
        logger.error(f"[FAIL] {param_file} not found. Ensure Foundation Optuna updated both JSONs.")
        sys.exit(1)

    with open(base_params_path, 'r') as f:
        base_params = json.load(f)
    
    logger.info(f"Loaded Foundation Arch ({param_file}): seq_len={base_params['seq_len']}, tcn={base_params['tcn_channels']}, lstm={base_params['lstm_hidden']}×{base_params['num_lstm_layers']}")

    logger.info(f"🔍 [Audit Diagnostic] foundation_val_dir: {foundation_val_dir.absolute()}")
    logger.info(f"🔍 [Audit Diagnostic] Directory exists? {foundation_val_dir.exists()}")
    
    val_files = sorted(list(foundation_val_dir.glob("*.parquet")))
    logger.info(f"🔍 [Audit Diagnostic] Found {len(val_files)} parquet files in {foundation_val_dir}")
    
    if not val_files:
        logger.error(f"[FAIL] NENHUM ARQUIVO .parquet ENCONTRADO EM {foundation_val_dir}!")
        raise FileNotFoundError(f"foundation_val_dir is empty: {foundation_val_dir}")

    # v10.11: Carregamento seletivo — carregamos apenas os Parquets que 
    # de fato possuem amostras no OOF, economizando MUITA RAM.
    logger.info(f"🔍 [Audit] Loading only necessary validation segments for Foundation inference...")
    
    # Identificamos os índices globais necessários no OOF
    needed_indices = df_oof["original_row_idx"].to_list()
    min_idx, max_idx = min(needed_indices), max(needed_indices)
    
    # Escaneamos os dados brutos e filtramos antecipadamente
    # (Supondo que a ordem dos arquivos em val_files bate com o row index global)
    dfs_val = []
    dfs_ctx = []
    current_global_offset = 0
    for i, vf in enumerate(val_files):
        # Scan rápido para saber o tamanho
        num_rows = pl.scan_parquet(vf).select(pl.len()).collect().item()
        
        file_start = current_global_offset
        file_end = current_global_offset + num_rows
        
        # Se este arquivo contém algum índice do OOF (considerando o lookback do seq_len)
        # Notas: precisamos carregar seq_len atrás para cada predição
        seq_len_req = base_params['seq_len']
        if not (file_end < (min_idx - seq_len_req) or file_start > max_idx):
            df_i = pl.read_parquet(vf, columns=feature_cols + ['target', 'island_id'])
            df_i = df_i.with_columns(pl.col('island_id') + (i * 10000))
            dfs_val.append(df_i)

            # Lazy Context Optimization: Escaneia APENAS o parquet correspondente a este arquivo do Validator Base
            ctx_file = Path(context_dir) / vf.name
            if ctx_file.exists():
                lf_c = pl.scan_parquet(str(ctx_file))
                # Através do offset, o índice bate exatamente com o Global Row Index esperado pelo Model OOF
                lf_c = lf_c.with_row_index("original_row_idx", offset=file_start)
                dfs_ctx.append(lf_c)
            else:
                logger.warning(f"⚠️ Context file missing for intersection: {vf.name}")
            
        current_global_offset += num_rows
    
    if not dfs_val or not dfs_ctx:
        logger.error("❌ Falha crítica: Nenhum dado de validação ou contexto casa com os índices do OOF.")
        sys.exit(1)
        
    df_fval = pl.concat(dfs_val)
    lf_ctx = pl.concat(dfs_ctx)
    logger.info(f"✅ Selective Foundation Val & Context LazyFrames loaded: {len(df_fval):,} dense rows (Optimization: RAM SAVE)")

    X_val_raw    = df_fval.select(feature_cols).to_numpy().astype(np.float32)
    y_val_raw    = df_fval.select('target').to_numpy().flatten().astype(np.int64)
    island_val   = df_fval.select('island_id').to_numpy().flatten()

    import joblib
    
    base_dir = get_drive_session_path("MODELOS", config)
    scaler_path = resolve_local_drive(Path(base_dir) / config['pipeline_paths']['scaler_foundation'])
    logger.info(f"🔑 [Audit Fix] Scaler carregado de: {scaler_path}")
    scaler_base = joblib.load(scaler_path)
    X_val_norm = scaler_base.transform(X_val_raw).astype(np.float32)

    seq_len      = base_params['seq_len']
    # Island-aware dataset: janelas cruzando gaps são descartadas
    dataset_base = SequenceDataset(X_val_norm, y_val_raw, island_val, seq_len)
    loader_base  = DataLoader(dataset_base, batch_size=2048, shuffle=False, num_workers=4)

    num_features = len(feature_cols)
    model_base = Hybrid_TCN_LSTM(
        num_features=num_features,
        seq_len=base_params['seq_len'],
        tcn_channels=base_params['tcn_channels'],
        lstm_hidden=base_params['lstm_hidden'],
        num_lstm_layers=base_params['num_lstm_layers'],
        num_classes=3,
        dropout=base_params['dropout'],
    ).to(DEVICE)

    logger.info(f"[OK] [Audit Fix] Usando Base Model: {active_base_path.name}")
    try:
        sd = torch.load(active_base_path, map_location=DEVICE, weights_only=True)
        model_base.load_state_dict(sd.get('model_state_dict', sd))
    except Exception as e:
        logger.error(f"[FAIL] Erro ao carregar state_dict do modelo base: {e}")
        sys.exit(1)

    logger.info("[MODEL] Generating Foundation Model probabilities over Foundation Val...")
    probs_base, _ = generate_predictions(model_base, loader_base, DEVICE)

    # SequenceDataset offsets: first valid prediction corresponds to original_row_idx = valid_indices[i] + seq_len - 1
    # This ensures alignment even when island-aware logic drops rows in the middle of a block.
    n_preds = len(probs_base)
    base_row_idx = dataset_base.valid_indices[:n_preds] + (seq_len - 1)

    df_base_probs = pl.DataFrame({
        "original_row_idx": base_row_idx,
        "base_prob_sell":   probs_base[:, 0],
        "base_prob_neu":    probs_base[:, 1],
        "base_prob_buy":    probs_base[:, 2],
    })

    # ── RAM-Safe Join: OOF (Dense) × Context (Lazy) ──────────────────────────
    logger.info("🔗 RAM-Safe joining OOF predictions × Foundation probs × Context features (Lazy)...")

    # Primeiro unimos os densos (OOF + Foundation Probs)
    df_signals = df_oof.join(df_base_probs, on="original_row_idx", how="inner")
    
    # Agora fazemos o Join Lazy com o Contexto de 89M
    # O Polars vai filtrar lf_ctx para conter apenas as 5.000 linhas do df_signals!
    df_joined_lf = lf_ctx.join(df_signals.lazy(), on="original_row_idx", how="inner")
    
    # Coletamos o resultado (apenas as linhas filtradas)
    df_joined = df_joined_lf.sort("original_row_idx").collect()

    logger.info(f"✅ Fused rows after LAZY join: {len(df_joined):,}")

    # v4.9: Explicit guard — if true_target is missing, fail early with a clear error
    if "true_target" not in df_joined.columns:
        logger.error(
            "❌ 'true_target' column not found in fused dataset after inner join. "
            "Check that full_oof.parquet was generated with the correct schema by run_kfold_specialist.py."
        )
        return

    df_fused = df_joined.to_pandas()
    df_fused = df_fused.rename(columns={"true_target": "true_target"})  # explicit

    _build_fused_and_save(df_fused, config, output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# MODO LEGADO (Holdout Especialista)
# ══════════════════════════════════════════════════════════════════════════════
def load_and_predict(config, val_dir, context_dir, output_dir):
    """
    Legacy Mode: loads the specialist validation directory and runs inference
    with the Foundation + Specialist models. Alignment is positional (seq_len offset).
    Used when kfold.enabled = false.
    """
    # ── Load Context Data ──────────────────────────────────────────────────────
    context_files = sorted(list(Path(context_dir).glob("*.parquet")))
    if not context_files:
        logger.error(f"❌ Context files not found in {context_dir}. Run auditor_preprocessing first.")
        return

    df_context_list = [pl.read_parquet(cf).to_pandas() for cf in context_files]
    full_context_df = pd.concat(df_context_list, ignore_index=True)

    # ── Load Raw Feature Data ───────────────────────────────────────────────────────
    feature_cols = config['model']['feature_names']
    val_files    = sorted(list(Path(val_dir).glob("*.parquet")))
    dfs_val      = []
    for i, vf in enumerate(val_files):
        df_i = pl.read_parquet(vf, columns=feature_cols + ['target', 'island_id'])
        df_i = df_i.with_columns(pl.col('island_id') + (i * 10000))
        dfs_val.append(df_i.to_pandas())
    df_val = pd.concat(dfs_val, ignore_index=True)

    seq_len     = config['optimization']['search_space']['seq_len'][0]
    X_val_raw   = df_val[feature_cols].to_numpy().astype(np.float32)
    y_val_raw   = df_val['target'].to_numpy().astype(np.int64)
    island_raw  = df_val['island_id'].to_numpy()

    # ── Normalization ──────────────────────────────────────────────────────────
    base_dir = get_drive_session_path("MODELOS", config)
    scaler_foundation_path  = Path(base_dir) / config['pipeline_paths']['scaler_foundation']
    scaler_specialized_path = Path(base_dir) / config['pipeline_paths']['scaler_specialized']
    logger.info(f"🔑 [Audit Fix] Scalers: foundation={scaler_foundation_path.name}, specialist={scaler_specialized_path.name}")

    with open(scaler_foundation_path, 'rb') as f:
        scaler_base = pickle.load(f)
    with open(scaler_specialized_path, 'rb') as f:
        scaler_spec = pickle.load(f)

    X_val_base_norm = scaler_base.transform(X_val_raw).astype(np.float32)
    X_val_spec_norm = scaler_spec.transform(X_val_raw).astype(np.float32)

    # Island-aware datasets: janelas cruzando gaps são descartadas
    dataset_base = SequenceDataset(X_val_base_norm, y_val_raw, island_raw, seq_len)
    dataset_spec = SequenceDataset(X_val_spec_norm, y_val_raw, island_raw, seq_len)

    loader_base = DataLoader(dataset_base, batch_size=2048, shuffle=False, num_workers=4)
    loader_spec = DataLoader(dataset_spec, batch_size=2048, shuffle=False, num_workers=4)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Selection logic for parameters (Legacy Mode)
    use_dir_strategy = config.get('training', {}).get('specialization_weights', {}).get('use_best_f1_dir', True)
    base_param_file = "best_dir_params.json" if use_dir_strategy else "best_params.json"
    
    base_params_path = Path("src/cloud/base_model/otimizacao") / base_param_file
    spec_params_path = Path("src/cloud/base_model/otimizacao") / "best_params_specialist.json"

    if not base_params_path.exists() or not spec_params_path.exists():
        logger.error(f"❌ {base_param_file} ou best_params_specialist.json não encontrados.")
        return

    with open(base_params_path, 'r') as f:
        base_params = json.load(f)
    with open(spec_params_path, 'r') as f:
        spec_params = json.load(f)

    num_features = len(feature_cols)

    model_base = Hybrid_TCN_LSTM(
        num_features=num_features, seq_len=base_params['seq_len'],
        tcn_channels=base_params['tcn_channels'], lstm_hidden=base_params['lstm_hidden'],
        num_lstm_layers=base_params['num_lstm_layers'], num_classes=3, dropout=base_params['dropout']
    ).to(DEVICE)

    model_spec = Hybrid_TCN_LSTM(
        num_features=num_features, seq_len=spec_params['seq_len'],
        tcn_channels=spec_params['tcn_channels'], lstm_hidden=spec_params['lstm_hidden'],
        num_lstm_layers=spec_params['num_lstm_layers'], num_classes=3, dropout=spec_params['dropout']
    ).to(DEVICE)

    base_dir = get_drive_session_path("MODELOS", config)
    base_path = Path(base_dir) / config['pipeline_paths']['best_tcn_lstm_model']
    spec_path = Path(base_dir) / config['pipeline_paths']['best_specialized_model']

    try:
        model_base.load_state_dict(torch.load(base_path, map_location=DEVICE))
        model_spec.load_state_dict(torch.load(spec_path, map_location=DEVICE))
    except KeyError:
        base_dict = torch.load(base_path, map_location=DEVICE)
        spec_dict = torch.load(spec_path, map_location=DEVICE)
        model_base.load_state_dict(base_dict.get('model_state_dict', base_dict))
        model_spec.load_state_dict(spec_dict.get('model_state_dict', spec_dict))

    logger.info("Generating Foundation Probabilities...")
    probs_base, targets_aligned = generate_predictions(model_base, loader_base, DEVICE)

    logger.info("Generating Specialist Probabilities...")
    probs_spec, _ = generate_predictions(model_spec, loader_spec, DEVICE)

    # Positional alignment (seq_len offset)
    aligned_context_df = full_context_df.iloc[seq_len - 1: len(full_context_df) - 1].reset_index(drop=True)

    if len(aligned_context_df) != len(targets_aligned):
        logger.warning(f"Feature alignment mismatch! Context: {len(aligned_context_df)}, Targets: {len(targets_aligned)}")
        min_len = min(len(aligned_context_df), len(targets_aligned))
        aligned_context_df = aligned_context_df.iloc[:min_len]
        probs_base        = probs_base[:min_len]
        probs_spec        = probs_spec[:min_len]
        targets_aligned   = targets_aligned[:min_len]

    df_fused = aligned_context_df.copy()
    df_fused["base_prob_sell"] = probs_base[:, 0]
    df_fused["base_prob_neu"]  = probs_base[:, 1]
    df_fused["base_prob_buy"]  = probs_base[:, 2]
    df_fused["spec_prob_sell"] = probs_spec[:, 0]
    df_fused["spec_prob_neu"]  = probs_spec[:, 1]
    df_fused["spec_prob_buy"]  = probs_spec[:, 2]
    df_fused["true_target"]    = targets_aligned

    _build_fused_and_save(df_fused, config, output_dir)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logger("auditor_labelling", "")
    conf = load_config()

    context_dir = conf['pipeline_paths'].get('auditor_context_dir', 'data/auditor/context')
    fused_dir   = conf['pipeline_paths'].get('fused_dataset_dir', 'data/auditor/dataset_fused')

    kfold_enabled = conf.get('pre_processing', {}).get('kfold', {}).get('enabled', False)

    if kfold_enabled:
        logger.info("🔁 K-Fold Mode detected (kfold.enabled=true) → loading full_oof.parquet")
        load_and_fuse_kfold(conf, context_dir, fused_dir)
    else:
        sell_th = conf['pre_processing']['labelling'].get('sell_threshold', 0.003)
        buy_th  = conf['pre_processing']['labelling'].get('buy_threshold', 0.003)
        mins    = conf['pre_processing']['labelling'].get('horizon_minutes', 15)
        spec_val_dir = Path(get_labelled_dir(conf)) / "val"

        if spec_val_dir.exists():
            logger.info(f"📂 Legado Mode: {spec_val_dir}")
            load_and_predict(conf, spec_val_dir, context_dir, fused_dir)
        else:
            logger.error(f"❌ {spec_val_dir} not found. Rode o pipeline novamente.")
