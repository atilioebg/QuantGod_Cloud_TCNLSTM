# 📒 8. Artifacts & Audit Guide — QuantGod v5.0

Este guia é a referência completa de todos os artefatos gerados pelo pipeline, tanto **localmente** quanto no **Google Drive**.

---

## 🗂️ Estrutura no Google Drive (por sessão de treino)

Cada execução do pipeline cria uma **pasta raiz única** no Drive, identificada pelos parâmetros de rotulação e o timestamp da sessão. Todos os artefatos daquela corrida ficam centralizados aqui, organizados de forma granular por camada do projeto.

```text
drive:PROJETOS/RESULTADOS_SELL_{S}_BUY_{B}_{M}min/
└── {timestamp}/                   # Única pasta por execução (ex: 08_03_2026_v000)
    ├── MODELOS/                   # Artefatos organizados (via transfer.py)
    │   ├── CONFIG/
    │   ├── BASE_MODEL/
    │   ├── SPECIALIST/
    │   ├── AUDITOR/
    │   └── REPORTS/
    ├── PRE_PROCESSED/             # Parquets sem target
    ├── LABELLED/                  # Parquets com target
    └── AUDITORIA/                 # Logs e Relatórios de QA (ETL/Labelling/KFold)
```

> [!NOTE]
> Os splits locais (`labelled/train`, `labelled/val`, `specialized/`) **não sobem para o Drive**. São dados derivados que podem ser sempre recriados a partir de `LABELLED` + `split_dataset.py`.

O sufixo e o timestamp são gerados automaticamente por `path_utils.get_drive_session_root()` com base nos parâmetros do `master_config.yaml`.

---

## 🗃️ Estrutura Local (na VM / RunPod)

```
data/L2/
├── temp_raw/               ← Downloads temporários de ZIPs brutos (auto-limpados)
├── pre_processed/          ← Saída do ETL: parquets sem target
├── labelled/               ← Saída do Labelling: parquets com coluna 'target'
│   ├── train/              ← 70% mais antigos (Modelo Base)
│   └── val/                ← 30% mais recentes (separados cronologicamente)
└── specialized/            ← Split do val da Fundação para o Especialista
    ├── train/              ← 80% do val da Fundação (Modelo Especialista)
    └── val/                ← 20% final (OOF puro - nunca visto por nenhum modelo)

data/auditor/
├── oof_predictions/        ← fold_0..4.parquet + full_oof.parquet
├── dataset_fused/          ← Features + Logits fundidos para o Auditor XGBoost
└── context/                ← Contexto de mercado

data/models/
├── best_tcn_lstm.pt        ← Melhor peso por F1 Macro (Foundation)
├── best_tcn_lstm_dir.pt    ← Melhor peso por F1 Direcional (Foundation)
├── scaler_foundation.pkl   ← StandardScaler fit no treino da Fundação
├── best_specialist.pt      ← Melhor peso do Especialista
├── scaler_specialized.pkl  ← StandardScaler fit no treino do Especialista
├── auditor_xgboost.json    ← Modelo XGBoost Auditor
└── scaler_auditor.pkl      ← StandardScaler do Auditor

logs/
├── etl/                    ← Logs do run_pipeline.py
├── labelling/              ← Logs do run_labelling.py
├── split_dataset/          ← Logs do split_dataset.py
├── optimization/           ← Logs do run_foundation.py (Optuna)
├── kfold_specialist/       ← Logs do run_kfold_specialist.py
└── specialization/         ← Logs do run_specialization.py
```

---

## 📋 Artefatos por Estágio

### 1️⃣ ETL — `run_pipeline.py`

| Artefato | Caminho Local | Destino no Drive |
| :--- | :--- | :--- |
| Log de operação | `logs/etl/etl_{ts}.log` | `AUDITORIA/ETL/` |
| Relatório de qualidade | `docs/reports/data_quality_report.json` | `AUDITORIA/ETL/` |
| Tabela de auditoria | `docs/reports/audit_summary.csv` | `AUDITORIA/ETL/` |
| Manifesto de arquivos pulados | `docs/reports/pipeline_skip_manifest.json` | `AUDITORIA/ETL/` |
| Dataset pré-processado | `data/L2/pre_processed/*.parquet` | `PRE_PROCESSED/` |

---

### 2️⃣ Labelling — `run_labelling.py`

| Artefato | Caminho Local | Destino no Drive |
| :--- | :--- | :--- |
| Log de rotulação | `logs/labelling/labelling_{ts}.log` | `AUDITORIA/LABELLING/` |
| Relatório QA pytest | `data/L2/labelled/labelling_health_QA.log` | `AUDITORIA/LABELLING/` |
| Dataset rotulado | `data/L2/labelled/*.parquet` | `LABELLED/` |

---

### 3️⃣ Split — `split_dataset.py`

| Artefato | Caminho Local | Destino no Drive |
| :--- | :--- | :--- |
| Log de split | `logs/split_dataset/split_dataset_{ts}.log` | `AUDITORIA/SPLIT/` |
| Sumário com hashes | `data/L2/split_summary.json` | `AUDITORIA/SPLIT/` |

> [!IMPORTANT]
> As pastas `labelled/train`, `labelled/val` e `specialized/` **não sobem para o Drive**.

---

### 4️⃣ Foundation Optuna — `run_foundation.py`

| Artefato | Caminho Local | Destino no Drive |
| :--- | :--- | :--- |
| Log de otimização | `logs/optimization/optimization_{ts}.log` | via `transfer.py` |
| Banco Optuna | `data/models/optuna_tcn_lstm.db` | `MODELOS/foundation/` |
| Melhores parâmetros (Macro) | `src/cloud/base_model/otimizacao/best_params.json` | `MODELOS/foundation/` |
| Melhores parâmetros (Dir) | `src/cloud/base_model/otimizacao/best_dir_params.json` | `MODELOS/foundation/` |
| Pesos do modelo (Macro) | `data/models/best_tcn_lstm.pt` | `MODELOS/foundation/` |
| Pesos do modelo (Dir) | `data/models/best_tcn_lstm_dir.pt` | `MODELOS/foundation/` |
| Scaler | `data/models/scaler_foundation.pkl` | `MODELOS/foundation/` |

---

### 5️⃣ K-Fold Specialist — `run_kfold_specialist.py`

| Artefato | Caminho Local | Destino no Drive |
| :--- | :--- | :--- |
| Log do K-Fold | `logs/kfold_specialist/kfold_specialist_{ts}.log` | `AUDITORIA/KFOLD_SPECIALIST/` |
| Relatório QA de segurança | `data/auditor/oof_predictions/kfold_security_QA.log` | `AUDITORIA/KFOLD_SPECIALIST/` |
| Predições OOF por fold | `data/auditor/oof_predictions/fold_{k}.parquet` | local apenas |
| OOF completo (cronológico) | `data/auditor/oof_predictions/full_oof.parquet` | local apenas |
| Scaler por fold | `data/auditor/oof_predictions/scaler_fold_{k}.pkl` | local apenas |

---

### 6️⃣ Auditor XGBoost — `train_xgboost.py`

| Artefato | Caminho Local | Destino no Drive |
| :--- | :--- | :--- |
| Log do Auditor | `logs/auditor/auditor_{ts}.log` | `AUDITORIA/AUDITOR/` |
| Feature Importance | `docs/reports/feature_importance.json` | `AUDITORIA/AUDITOR/` |
| Modelo XGBoost | `data/models/auditor_xgboost.json` | `MODELOS/specialized/` |
| Scaler do Auditor | `data/models/scaler_auditor.pkl` | `MODELOS/specialized/` |

---

## 🛡️ Tags de Auditoria (Gold v4.6+)

| Tag | Significado |
| :--- | :--- |
| `[DNN_INPUT]` | Feature usada como entrada do TCN-LSTM (coluna de feature) |
| `[XGB_ONLY]` | Logit gerado por um modelo (entrada exclusiva do Auditor XGBoost) |
| `[RAW_DATA]` | Dado bruto de suporte do Order Book (não usado como feature DNN) |
| `🧟 DEAD FEATURE` | Feature sem variância — sinal de bug ou feed de dados travado |
| `⚠️ HIGH TAIL` | Outlier extremo detectado (Z-Score > 12) |
| `✅ VALID` | Arquivo aprovado na validação de integridade |
| `🔧 FIXED` | Arquivo com gap temporal que foi curado pelo protocolo Island Split |
| `❌ INVALID` | Arquivo rejeitado — não salvo no pre_processed |
