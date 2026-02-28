# 📒 8. Artifacts & Audit Guide — QuantGod v4.6 Gold

Este guia serve como uma referência rápida para localizar todos os logs e relatórios gerados pelo pipeline em tempo de execução.

---

## 🏗️ 1. Estágio: ETL & Pré-Processamento
**Script**: `run_pipeline.py`

| Artefato | Caminho (Path) | Descrição |
| :--- | :--- | :--- |
| **Log de Operação** | `logs/etl/etl_YYYYMMDD_HHMMSS.log` | Registro detalhado de download, streaming e clipping. |
| **Métricas de Sanidade** | `docs/reports/data_quality_report.json` | **[CRÍTICO]** Resumo JSON de integridade, linhagem e anomalias. |
| **Dataset Parquet** | `data/L2/pre_processed_L2/*.parquet` | Features prontas (831 colunas: 30 Features Gold + 800 Raw). |

---

## 🏷️ 2. Estágio: Labelling & Data Split
**Scripts**: `run_labelling.py`, `split_dataset.py`

| Artefato | Caminho (Path) | Descrição |
| :--- | :--- | :--- |
| **Log de Labelling** | `logs/labeling/labeling_YYYYMMDD_HHMMSS.log` | Status do processamento paralelo de rótulos. |
| **Relatório de Classes** | `docs/reports/labelling_audit.json` | Distribuição de sinais BUY/SELL/NEUTRAL. |
| **Summary de Split** | `data/splits/split_summary.json` | Prova de isolamento temporal (Purge Gap) e K-Fold. |

---

## 🧠 3. Estágio: Treino & Otimização
**Scripts**: `run_optuna.py`, `run_training.py`, `train_xgboost.py`

| Artefato | Caminho (Path) | Descrição |
| :--- | :--- | :--- |
| **Banco Optuna** | `sqlite:///optuna_tcn_lstm_v1_finetune.db` | Histórico completo de todos os trials e hiperparâmetros. |
| **Melhores HPs** | `src/cloud/base_model/configs/best_params.json` | Parâmetros vencedores injetados no treino. |
| **Pesos do Modelo** | `data/models/*.pt` (TCN) / `*.json` (XGB) | O cérebro da IA e do Juiz Auditor final. |
| **Scalers** | `data/models/scaler_*.pkl` | Normalizadores cruciais para live inference. |

---

## 🛡️ 4. Guia de Tags de Auditoria (v4.6)
Ao ler os logs ou JSONs, atente-se a estas tags:
*   `[DNN_INPUT]`: Refere-se às 30 features obrigatórias do modelo.
*   `[XGB_ONLY]`: Refere-se aos 6 logits de cada modelo (Base e Especialista).
*   `[RAW_DATA]`: Refere-se aos logs brutos do Order Book (suporte).
*   `🧟 DEAD FEATURE`: Feature sem variância (coluna estática).
*   `⚠️ HIGH TAIL`: Outliers extremos (Z-Score > 12).
