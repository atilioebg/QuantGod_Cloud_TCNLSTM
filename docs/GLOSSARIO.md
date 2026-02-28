# 📖 Glossário de Ativos — QuantGod Cloud ⚡

Este documento serve como um mapa central para localizar e entender a finalidade de cada arquivo gerado ou utilizado no ecossistema QuantGod.

---

## 🤖 1. Modelos e Pesos (`models/`)
Arquivos binários contendo a inteligência treinada e os normalizadores de dados.

| Arquivo/Pasta | Formato | Descrição |
|:---|:---|:---|
| `best_tcn_lstm.pt` | `.pt` (PyTorch) | **Modelo Fundação (Base)**: Pesos da rede TCN+LSTM balanceada. |
| `best_specialized_model.pt` | `.pt` (PyTorch) | **Modelo Especialista**: Pesos afinados para Sniper (SELL/BUY). |
| `auditor_xgboost.json` | `.json` (XGBoost) | **Auditor (Juiz)**: Modelo de regressão/classificação de confiabilidade. |
| `scaler_foundation.pkl` | `.pkl` (Pickle) | Normalizador (StandardScaler) das features OHLC do modelo base. |
| `scaler_specialized.pkl` | `.pkl` (Pickle) | Normalizador das features de fluxo (RobustScaler) do especialista. |
| `audit_scaler.pkl` | `.pkl` (Pickle) | Normalizador das meta-features injetadas no Auditor. |

---

## 📄 2. Documentação Técnica (`docs/`)
Manuais e relatórios de certificação.

| Arquivo | Propósito |
|:---|:---|
| `docs/0_REPO_MAP.md` | Estrutura de pastas, fluxogramas e dependências. |
| `docs/5_MODEL_ARCHITECTURE.md` | Detalhes matemáticos das redes TCN, LSTM e XGBoost. |
| `docs/reports/Relatorio_Auditoria_v4.5_Final.md` | **Certificação OOF**: Prova de zero-leakage e estabilidade para investidores. |
| `docs/6_OPERATIONAL_MANUAL.md` | Guia passo a passo para execução em nuvem (RunPod). |
| `README.md` (Raiz) | Overview, Quick Start e guia de instalação. |

---

## 📊 3. Dados e Snapshots (`data/`)
Armazenamento de dados em diferentes estágios de processamento.

| Pasta/Arquivo | Formato | Conteúdo |
|:---|:---|:---|
| `data/L2/raw/` | `.zip` | Dados brutos de Order Book (Level 2) direto da exchange. |
| `data/L2/pre_processed/` | `.parquet` | Dados após ETL: 833 colunas, amostragem de 1 min. |
| `data/L2/labelled_*/` | `.parquet` | Datasets com a coluna `target` (0, 1, 2) calculada. |
| `data/audit_output/snapshots/` | `.csv` | Amostras rápidas (Snapshots) para conferência humana de cada etapa. |
| `full_oof.parquet` | `.parquet` | Predições **Out-of-Fold**: Base para treinamento do Auditor sem vazamento. |

---

## 📜 4. Logs e Rastreabilidade (`logs/`)
Registros de execução para depuração e auditoria de treinamento.

| Pasta | Conteúdo |
|:---|:---|
| `logs/kfold_specialist/` | Histórico de loss e métricas (F1) por fold do Especialista. |
| `logs/drive_validation/` | Logs de integridade do sync Rclone entre Local e Cloud. |
| `logs/tests/` | Saída detalhada da suite de testes Pytest. |

---

## ⚙️ 5. Configurações e Métricas de Auditoria (`configs/`)
A "Fonte Única de Verdade" para hiperparâmetros e execução.

| Arquivo/Métrica | Formato | Uso |
|:---|:---|:---|
| `master_config.yaml` | `.yaml` | Configuração mestra: thresholds, features, janelas e paths. |
| **`max_idx_gap`** | Float (min) | Auditoria: Maior distância entre timestamps consecutivos (Alvo: 1.0). |
| **`max_vol_gap`** | Float (min) | Auditoria: Maior intervalo contínuo com zero trades (`tick_count=0`). |
| `pytest.ini` | `.ini` | Flags de execução dos testes automatizados. |
| `rclone.conf` | `.conf` | Chaves e tokens para acesso ao Google Drive Cloud. |

---

> **Dica**: Sempre consulte o `master_config.yaml` antes de iniciar qualquer etapa, pois ele define quais pastas de dados o sistema buscará.
