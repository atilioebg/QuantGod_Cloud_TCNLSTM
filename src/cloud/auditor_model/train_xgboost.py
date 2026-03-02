"""
train_xgboost.py — XGBoost Auditor Model Training (Meta-Labeling Juiz)

Este script treina o Auditor a partir do dataset fundido (Logits + Contexto).
O alvo é o `meta_target` (1 = DNN Acertou, 0 = DNN Errou).
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_score
import yaml
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import sys

project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

# Features agora são ingeridas dinamicamente do master_config
# A lista de features aguardadas se encontra em `model.auditor_features`

def load_config():
    aud_cfg_path = Path("src/cloud/auditor_model/configs/auditor_config.yaml")
    if aud_cfg_path.exists():
        with open(aud_cfg_path) as f:
            aud_cfg = yaml.safe_load(f)
    else:
        # Fallback minimal config
        aud_cfg = {'xgboost': {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}}
    return aud_cfg

def load_data(fused_dir: str):
    logger.info(f"Carregando dados fundidos de {fused_dir}...")
    
    train_dir = Path(fused_dir) / "train"
    val_dir = Path(fused_dir) / "val"
    
    train_files = list(train_dir.glob("*.parquet"))
    val_files = list(val_dir.glob("*.parquet"))
    
    if not train_files or not val_files:
        raise FileNotFoundError(f"Arquivos .parquet não encontrados nas pastas de split em {fused_dir}")
        
    df_train = pl.concat([pl.read_parquet(f) for f in train_files]).to_pandas()
    df_val = pl.concat([pl.read_parquet(f) for f in val_files]).to_pandas()
    
    return df_train, df_val

def train_auditor():
    aud_cfg = load_config()
    
    # Adicionando consumo do master config para paths coerentes
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        master_cfg = yaml.safe_load(f)
        
    xgb_params = aud_cfg.get('xgboost', {})
    
    
    fused_dir = master_cfg['pipeline_paths'].get('fused_dataset_dir', "data/auditor/dataset_fused")
    df_train, df_val = load_data(fused_dir)
    
    # Busca a lista oficial de features do XGBoost cadastrada no yaml
    xgb_features = master_cfg['model'].get('auditor_features', [])
    if not xgb_features:
        logger.error("❌ 'auditor_features' não definidas no master_config.yaml sob o node 'model'.")
        sys.exit(1)
        
    X_train = df_train[xgb_features].to_numpy().astype(np.float32)
    y_train = df_train['meta_target'].to_numpy().astype(np.int64)
    
    X_val = df_val[xgb_features].to_numpy().astype(np.float32)
    y_val = df_val['meta_target'].to_numpy().astype(np.int64)
    
    # NOVO: Fit e Transform Scaler (Exigência do Pipiline QuantGod)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    
    logger.info(f"Tamanho Treino: {len(X_train)} | Tamanho Val: {len(X_val)}")
    logger.info(f"Ocorrência de Acertos (1) no Treino: {y_train.mean():.2%}")
    logger.info(f"Ocorrência de Acertos (1) no Val: {y_val.mean():.2%}")
    
    # Balanceamento: Weight para a classe 1 (Acerto). Se Erros forem raros, compensa.
    # Ex: ratio = qty_erros / qty_acertos
    scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
    # Se scale_pos_weight > 1, penaliza erro no falso negativo. Se < 1, penaliza falso positivo.
    # Queremos evitar falsos positivos do Meta_Target (achar que vai acertar e errar). 
    # Então queremos um modelo conservador.
    
    # v4.9: Correct GPU detection for XGBoost (torch.cuda.is_available() is reliable)
    import torch as _torch
    _xgb_device = 'cuda' if _torch.cuda.is_available() else 'cpu'

    model = xgb.XGBClassifier(
        n_estimators=xgb_params.get('n_estimators', 300),
        max_depth=xgb_params.get('max_depth', 5),
        learning_rate=xgb_params.get('learning_rate', 0.05),
        subsample=xgb_params.get('subsample', 0.8),
        colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        device=_xgb_device,
        random_state=42
    )
    
    logger.info("Iniciando treinamento do XGBoost Meta-Labeler...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    
    logger.info("Treinamento finalizado. Gerando métricas de validação...")
    
    # ── Avaliação Dinâmica de Threshold ───────────────────────────────────────
    # Probabilidades de SER UM ACERTO (meta_target = 1)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    use_dyn = master_cfg['model']['auditor'].get('use_dynamic_threshold', True)
    fallback_th = master_cfg['model']['auditor'].get('manual_threshold', 0.65)
    
    if use_dyn:
        logger.info("=== Calibração de Threshold do Auditor (F-Beta 0.5) ===")
        best_threshold = 0.5
        best_fbeta = 0.0
        
        # Maximizando Beta = 0.5 (peso maior na precisão para evitar veto falso positivo)
        beta = 0.5
        beta_sq = beta ** 2
        
        for t in np.arange(0.50, 0.95, 0.02):
            y_pred_t = (y_pred_proba >= t).astype(int)
            acc_t = accuracy_score(y_val, y_pred_t)
            prec_t = precision_score(y_val, y_pred_t, zero_division=0)
            
            # Recall customizado via class report
            rep = classification_report(y_val, y_pred_t, output_dict=True, zero_division=0)
            rec_t = rep['1']['recall'] if '1' in rep else 0.0
            
            fbeta_t = (1 + beta_sq) * (prec_t * rec_t) / ((beta_sq * prec_t) + rec_t + 1e-9)
            
            logger.info(f"Threshold {t:.2f} -> Prec: {prec_t:.4f} | Rec: {rec_t:.4f} | F-0.5: {fbeta_t:.4f}")
            
            if fbeta_t > best_fbeta:
                best_fbeta = fbeta_t
                best_threshold = t
                
        logger.info(f"🚀 Melhor Threshold Automático Escolhido (F-0.5 Máximo): {best_threshold:.2f} (F: {best_fbeta:.4f})\n")
        threshold = best_threshold
    else:
        threshold = fallback_th
        logger.info(f"=== Fallback Threshold Manual do master_config.yaml: {threshold:.2f} ===")
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_pred_proba)
    
    logger.info(f"=== Performance Final do XGBoost Auditor (Threshold: {threshold:.2f}) ===")
    logger.info(f"Accuracy: {acc:.4f} | F1-Score (Acertos): {f1:.4f} | ROC-AUC: {roc:.4f}")
    logger.info(f"Relatório de Classificação:\n{classification_report(y_val, y_pred, target_names=['ERRO (0)', 'ACERTO (1)'])}")
    
    # ── Feature Importance ───────────────────────────────────────────────────
    importance = model.feature_importances_
    imp_ranked = sorted(zip(xgb_features, importance), key=lambda x: -x[1])
    logger.info("Feature Importance (Top 10):")
    for name, imp in imp_ranked[:10]:
        logger.info(f"  {name:20s}: {imp:.4f}")
        
    # ── Exibição: Quadra Final de Inferência (Amostragem) ────────────────────
    logger.info(" " * 60)
    logger.info("=== Output Quádruplo Final (Amostra do Val Set) ===")
    
    labels_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}
    
    # Extrair campos brutos
    spec_probs = df_val[['spec_prob_sell', 'spec_prob_neu', 'spec_prob_buy']].values
    
    samples = np.random.choice(len(df_val), size=10, replace=False)
    for idx in samples:
        # Recuperar label predição DNN (Spec)
        probs = spec_probs[idx]
        dnn_class = np.argmax(probs)
        dnn_label = labels_map[dnn_class]
        dnn_score = probs[dnn_class]
        
        xgb_proba = y_pred_proba[idx]
        xgb_status = "Executa" if xgb_proba >= threshold else "Nao_Executa"
        
        # Real
        true_class = df_val.iloc[idx]['true_target']
        is_correct = (dnn_class == true_class)
        
        logger.info(
            f"🎯 [Label: {dnn_label:7s} | Score_DNN: {dnn_score:.2f} | "
            f"Status_XGB: {xgb_status:11s} | Proba_XGBoost: {xgb_proba:.2f}]  --> Realidade: {'Acertou' if is_correct else 'Errou'}"
        )
        
    logger.info(" " * 60)
    
    # ── Salvamento ───────────────────────────────────────────────────────────
    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(master_cfg['pipeline_paths'].get('auditor_model', 'data/models/auditor_xgboost.json'))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"🤖 Modelo XGBoost Auditor Salvo: {model_path}")
    
    scaler_path = Path(master_cfg['pipeline_paths'].get('scaler_auditor', 'data/models/scaler_auditor.pkl'))
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"💾 Scaler XGBoost Auditor Salvo: {scaler_path}")


if __name__ == "__main__":
    setup_logger("train_xgboost", "")
    train_auditor()
