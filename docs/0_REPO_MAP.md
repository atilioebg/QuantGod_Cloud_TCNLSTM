# 🗺️ Repo Map

> **Repo:** [`atilioebg/QuantGod_Cloud_TCNLSTM`](https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM) | **Branch:** `main` | Updated: 2026-03-18

---

## Source Code (`src/cloud/`)

### Base Model (`src/cloud/base_model/`)

| Arquivo | Propósito |
|:---|:---|
| `configs/master_config.yaml` | **Fonte única de verdade** — Configurações de ETL, Labelling, Treino e Auditoria. |
| `labelling/run_labelling.py` | **Motor Prado-IID** — Triple Barrier Labelling com Volatilidade Adaptativa (EWMA). |
| `models/model.py` | `Hybrid_TCN_LSTM` — CausalConv1D stack + LSTM + MLP head. |
| `pre_processamento/etl/extract.py` | Streaming de ZIPs (ob200/ob500) do GDrive/local via Polars. |
| `pre_processamento/etl/merge_l2_trades.py` | Join Atomic (join_asof) — Sincronização de milissegundos entre L2 e Trades. |
| `pre_processamento/etl/event_sampler.py` | **Coração do ETL v5.0** — Geração de barras IID (Dollar, Tick, Information). |
| `pre_processamento/etl/transform.py` | Facade para cálculo de Micro-Price, OFI, Slope, RDI e Features Sniper. |
| `pre_processamento/orchestration/run_pipeline.py` | Orquestrador unificado de processamento paralelo por dia. |
| `treino/run_training.py` | Loop de treino: AdamW + CosineAnnealingLR + EarlyStopping (F1 Macro). |
| `auditoria/visualize_labels.py` | Ferramenta de visualização gráfica dos sinais Triple Barrier sobre os candles. |

### Auditor Model (`src/cloud/auditor_model/`)

| Arquivo | Propósito |
|:---|:---|
| `configs/auditor_config.yaml` | `n_folds=5`, XGBoost params, paths (base model checkpoint, xgb output). |
| `feature_engineering_meta.py` | 14 meta-features a partir do output do base model — **sem warm-up**. |
| `train_xgboost.py` | OOF Walk-Forward K=5 com `TimeSeriesSplit` — zero data leakage. |
| `binance_adapter.py` | Binance Futures WS + REST sync para inferência live e stream de dados. |

---

## Dados (`data/`)

| Caminho | Conteúdo |
|:---|:---|
| `data/L2/pre_processed/` | Parquets IID — 833 colunas, linhas variáveis por atividade de mercado. |
| `data/L2/labelled_*/` | Parquets rotulados — Triple Barrier (0, 1, 2) + 833 colunas. |
| `data/models/` | `base_model.pt`, `scaler_finetuning.pkl`, `xgb_auditor.json`. |
| `data/live/` | Buffers de papel trading e inferência em tempo real. |

---

## Testes (`tests/`)

| Arquivo | Tipo | Propósito |
|:---|:---|:---|
| `test_model.py` | Unitário | `Hybrid_TCN_LSTM`: shapes, causal conv, determinismo, gradientes. |
| `test_meta_features.py` | Unitário | Auditor indicators (RSI, EMA, Bollinger, ATR, Entropy). |
| `test_cloud_etl_output.py` | Dados | 833 cols, book sorted, no NaN, chronological. |
| `tests/etl/test_advanced_sampler.py` | Lógica | Validação de Reconciliação de Transbordo (Dollar/Tick Bars). |
| `tests/labelling/test_tri_barrier_logic.py` | Lógica | Validação matemática do Triple Barrier e EWMA Volatility. |

---

## Documentação (`docs/`)

| Arquivo | Propósito |
|:---|:---|
| [`0_REPO_MAP.md`](0_REPO_MAP.md) | Este arquivo. |
| [`GLOSSARIO.md`](GLOSSARIO.md) | Terminologia técnica (IID Bars, Triple Barrier, OBI). |
| [`3_DATA_ENGINEERING.md`](3_DATA_ENGINEERING.md) | Detalhes do ETL v5.0 e amostragem de eventos. |
| [`3_ETL_GOLD_STANDARD_v5.0.md`](3_ETL_GOLD_STANDARD_v5.0.md) | **Audit Layers v5.0** — Critérios de Aprovação do Domínio da Informação. |
| [`4_LABELING_STRATEGY.md`](4_LABELING_STRATEGY.md) | Filosofia Prado-IID e barreiras adaptativas. |
| [`5_MODEL_ARCHITECTURE.md`](5_MODEL_ARCHITECTURE.md) | Arquitetura Hybrid TCN-LSTM e Auditor XGBoost. |
| [`7_DATA_REFERENCE.md`](7_DATA_REFERENCE.md) | Referência técnica completa das 32 features. |
ório Gold Standard** | `docs/reports/Relatorio_Auditoria_v4.5_Final.md` | `audit_v45.py` |
