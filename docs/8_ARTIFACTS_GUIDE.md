# 📒 8. Artifacts & Audit Guide — QuantGod v4.6 Gold

Este guia serve como uma referência rápida e exaustiva para localizar todos os artefatos (logs, JSONs, modelos e datasets) gerados pelo pipeline em tempo de execução.

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
| **Log de Labelling** | `logs/labelling/labelling_YYYYMMDD_HHMMSS.log` | Status do processamento paralelo de rótulos. |
| **Relatório de Classes** | `docs/reports/labelling_audit.json` | Distribuição de sinais BUY/SELL/NEUTRAL. |
| **Summary de Split** | `data/splits/split_summary.json` | Prova de isolamento temporal (Purge Gap) e K-Fold. |

---

## 🧠 3. Estágio: Treino & Otimização
**Scripts**: `run_optuna.py`, `run_training.py`, `train_xgboost.py`

### A. Metadados e Logs
| Artefato | Caminho (Path) | Descrição |
| :--- | :--- | :--- |
| **Banco Optuna** | `sqlite:///optuna_tcn_lstm_v1_finetune.db` | Histórico completo de todos os trials e hiperparâmetros. |
| **Logs Optuna** | `logs/optimization/optuna_*.log` | Detalhes técnicos de cada trial de otimização. |
| **Logs Treino** | `logs/training/train_*.log` | Métricas por época (Loss, F1, Accuracy). |
| **Melhores HPs** | `src/cloud/base_model/configs/best_params.json` | Parâmetros vencedores injetados no treino final. |

### B. Modelos Digitais & Sinergia (Pesos + Scalers)
> [!IMPORTANT]
> Modelos e Scalers são tratados como uma única unidade lógica de inferência.

| Componente | Pesos (Model Weights) | Normalizador (Scaler) | Descrição |
| :--- | :--- | :--- | :--- |
| **Fundação** | `data/models/best_tcn_lstm.pt` | `data/models/scaler_foundation.pkl` | Treinado em todo o dataset (Generalista). |
| **Especialista** | `data/models/best_tcn_lstm_dir.pt` | `data/models/scaler_specialized.pkl` | Foco em Sniper de Direção. |
| **Auditor** | `data/models/auditor_xgboost.json` | `data/models/scaler_auditor.pkl` | O "Juiz" final da operação. |

### C. Auditoria de Governança Gold v4.6
| Artefato | Caminho (Path) | Descrição |
| :--- | :--- | :--- |
| **Resumo Executivo** | `docs/reports/audit_summary.csv` | **Tabela mestre** com status `VALID`, `INVALID` ou `FIXED`. 🆕<br>Colunas: `status`, `max_gap_before`, `max_gap_after`, `features_healed`, `healing_details`. |
| **Quality Report** | `docs/reports/data_quality_report.json` | JSON detalhado com anomalias de cauda (Z-Score) e linhagem. |
| **Logs ETL** | `logs/etl/*.log` | Rastro técnico do processamento diário. |

### C. Metadados e Logs de Treino
| Artefato | Caminho (Path) | Descrição |
| :--- | :--- | :--- |
| **Banco Optuna** | `sqlite:///optuna_tcn_lstm_v1_finetune.db` | Histórico de trials e hiperparâmetros. |
| **Logs Optuna** | `logs/optimization/optuna_*.log` | Detalhes técnicos de cada trial. |
| **Logs Treino** | `logs/training/train_*.log` | Métricas por época (Loss, F1, Accuracy). |
| **Melhores HPs** | `src/cloud/base_model/configs/best_params.json` | Parâmetros vencedores do treino final. |

### D. Dados Intermediários de Auditoria
| Artefato | Caminho (Path) | Descrição |
| :--- | :--- | :--- |
| **OOF Predictions** | `data/auditor/oof_predictions/*.parquet` | Predições fora-da-amostra usadas para treinar o Auditor sem leakage. |
| **Fused Dataset** | `data/auditor/dataset_fused/*.parquet` | Dataset final consolidado (Features + Logits). |

---

## 🛡️ 4. Guia de Tags de Auditoria (v4.6 Gold)
*   `[DNN_INPUT]`: As 30 features do modelo TCN-LSTM.
*   `[XGB_ONLY]`: Os 6 logits (Buy/Sell/Neu) gerados por cada modelo.
*   `[RAW_DATA]`: Dados brutos de suporte do Order Book.
*   `🧟 DEAD FEATURE`: Feature sem variância (bug ou feed travado).
*   `⚠️ HIGH TAIL`: Outliers extremos (Z-Score > 12).
