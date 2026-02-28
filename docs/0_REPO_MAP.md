# 🗺️ Repo Map

> **Repo:** [`atilioebg/QuantGod_Cloud_TCNLSTM`](https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM) | **Branch:** `main` | Updated: 2026-02-20

---

## Source Code (`src/cloud/`)

### Base Model (`src/cloud/base_model/`)

| Arquivo | Propósito |
|:---|:---|
| `configs/base_model_config.yaml` | **Fonte única de verdade** — `class_weights`, `feature_names` (32), `seq_len`, `num_classes` |
| `labelling/run_labelling.py` | Gera labels SELL(0)/NEUTRAL(1)/BUY(2) via rolling_sum de log_ret_close |
| `labelling/labelling_config.yaml` | Parâmetros de rotulagem: `lookahead=60`, thresholds, paths de I/O |
| `models/model.py` | `Hybrid_TCN_LSTM` — CausalConv1D stack + LSTM + MLP head |
| `otimizacao/run_optuna.py` | Busca de hiperparâmetros (F1 Macro) com OOM guard via TrialPruned |
| `otimizacao/optimization_config.yaml` | Search space: `tcn_channels`, `lstm_hidden`, `lr`, `dropout`, `batch_size`, `seq_len` |
| `pre_processamento/etl/extract.py` | Leitura recursiva de ZIPs (ob200 + ob500) do GDrive/local |
| `pre_processamento/etl/transform.py` | Reconstrução do L2 book → Hard Cut 200 → 1s sampling → 1min resample → 32 features |
| `pre_processamento/etl/load.py` | Serialização para Parquet com compressão Snappy |
| `pre_processamento/etl/validate.py` | NaN, Inf, ordem temporal, gap detection |
| `pre_processamento/orchestration/run_pipeline.py` | Orquestrador do ETL — parallelismo por arquivo |
| `pre_processamento/configs/cloud_config.yaml` | Paths (rclone_mount, output_dir), sampling_interval_ms, ob_levels |
| `treino/run_training.py` | Loop de treino: AdamW + CosineAnnealingLR + EarlyStopping (F1 Macro) |
| `treino/training_config.yaml` | HPs finais (pós-Optuna): `batch_size`, `lr`, `dropout`, `seq_len`, paths |
| `auditoria/audit_v45.py` | **Protocolo Gold Standard v4.5** — E2E Audit, Snapshot, Anti-Leakage & Fusion |
| `auditoria/audit_v44.py` | Pipeline de Auditoria Legado (v4.3/v4.4) |

### Auditor Model (`src/cloud/auditor_model/`)

| Arquivo | Propósito |
|:---|:---|
| `configs/auditor_config.yaml` | `n_folds=5`, XGBoost params, paths (base model checkpoint, xgb output) |
| `feature_engineering_meta.py` | 14 meta-features a partir do output do base model — **sem warm-up** |
| `train_xgboost.py` | OOF Walk-Forward K=5 com `TimeSeriesSplit` — zero data leakage |
| `binance_adapter.py` | Binance Futures WS + REST sync (`U/u/lastUpdateId`) para inferência live |

---

## Dados (`data/`)

| Caminho | Conteúdo |
|:---|:---|
| `data/L2/pre_processed/` | Parquets ETL — 833 colunas, ~1440 linhas/dia |
| `data/L2/labelled_*/` | Parquets rotulados — 833 colunas + `target` ∈ {0,1,2} |
| `data/models/` | `base_model.pt`, `scaler_finetuning.pkl`, `xgb_auditor.json` |
| `data/live/` | Buffers de candles live do `binance_adapter.py` |

Consulte [`data/README.md`](../data/README.md) para detalhes de tamanhos e experimentos.

---

## Testes (`tests/`)

| Arquivo | Tipo | Propósito |
|:---|:---|:---|
| `conftest.py` | Fixtures | Constantes globais e fixtures sintéticas compartilhadas |
| `test_model.py` | Unitário | `Hybrid_TCN_LSTM`: shapes, simplex, causal conv, determinismo, gradientes |
| `test_meta_features.py` | Unitário | `feature_engineering_meta.py`: indicadores (RSI, EMA, Bollinger, ATR, Entropy) e `extract_meta_features` |
| `test_config_integrity.py` | Config | 4 YAMLs + consistência cross-config (`labelled_dir` deve coincidir) |
| `test_cloud_etl_output.py` | Dados | 833 cols, book sorted, no NaN, chronological |
| `test_preprocessed_quality.py` | Dados | Contagem de linhas, continuidade de datas, schema, nulls |
| `test_labelling_output.py` | Dados | Schema `target`, {0,1,2}, ≥2 classes/arquivo, balance global ≥3% |

Consulte [`tests/README.md`](../tests/README.md) para comandos e descrição detalhada de cada suite.

---

## Documentação (`docs/`)

| Arquivo | Propósito |
|:---|:---|
| [`0_REPO_MAP.md`](0_REPO_MAP.md) | Este arquivo |
| [`GLOSSARIO.md`](GLOSSARIO.md) | Dicionário de arquivos (logs, modelos, dados) |
| [`1_SETUP_AND_ENV.md`](1_SETUP_AND_ENV.md) | Hardware, CUDA, RunPod setup |
| [`2_DATA_COLLECTION.md`](2_DATA_COLLECTION.md) | Processo de obtenção dos dados Bybit L2 |
| [`3_DATA_ENGINEERING.md`](3_DATA_ENGINEERING.md) | Detalhes do ETL e feature engineering |
| [`4_LABELING_STRATEGY.md`](4_LABELING_STRATEGY.md) | Lógica de labelling e thresholds |
| [`5_MODEL_ARCHITECTURE.md`](5_MODEL_ARCHITECTURE.md) | **Referência primária** — arquitetura completa, constraints, execução |
| [`6_OPERATIONAL_MANUAL.md`](6_OPERATIONAL_MANUAL.md) | Guia de execução RunPod |
| [`7_DATA_REFERENCE.md`](7_DATA_REFERENCE.md) | Referência técnica completa dos dados |

---

## Raiz do Projeto

| Arquivo | Propósito |
|:---|:---|
| `README.md` | Overview geral + links para toda a documentação |
| `requirements.txt` | Dependências Python do projeto |
| `pytest.ini` | Configuração do pytest (testpaths, addopts) |
| `.gitignore` | Exclui venv, data, models, logs, tokens sensíveis |
| `rclone.conf` | Config de montagem do GDrive (Windows local) |
| `logs/LOGS_README.md` | Documentação das convenções de logs |

---

## Outputs Conhecidos (após pipeline completo)

| Artefato | Localização | Produzido por |
|:---|:---|:---|
| Parquets pré-processados | `data/L2/pre_processed/` | `run_pipeline.py` |
| Parquets rotulados | `data/L2/labelled_SELL_0004_BUY_0008_1h/` | `run_labelling.py` |
| Estudo Optuna | `data/models/optuna_study.db` | `run_optuna.py` |
| Base model checkpoint | `data/models/base_model.pt` | `run_training.py` |
| StandardScaler | `data/models/scaler_finetuning.pkl` | `run_training.py` |
| XGBoost Auditor | `data/models/xgb_auditor.json` | `train_xgboost.py` |
| **Relatório Gold Standard** | `docs/reports/Relatorio_Auditoria_v4.5_Final.md` | `audit_v45.py` |
