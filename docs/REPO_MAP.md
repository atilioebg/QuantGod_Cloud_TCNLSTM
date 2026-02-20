# 🗺️ Repo Map

> **Branch:** `tcn_lstm` | Updated: 2026-02-20

## Source Code (`src/cloud/`)

### Base Model (`src/cloud/base_model/`)

| File | Purpose |
|:---|:---|
| `configs/base_model_config.yaml` | **Single source of truth** — class_weights, feature_names, seq_len |
| `labelling/run_labelling.py` | Generates SELL/NEUTRAL/BUY labels from lookahead returns |
| `labelling/labelling_config.yaml` | Lookahead=60, thresholds: Short=-0.4%, Long=+0.8% |
| `models/model.py` | `Hybrid_TCN_LSTM` — CausalConv1D + LSTM + MLP head |
| `otimizacao/run_optuna.py` | Hyperparameter search (TCN+LSTM) with OOM guard |
| `otimizacao/optimization_config.yaml` | Search space: tcn_channels, lstm_hidden, lr, dropout |
| `pre_processamento/etl/extract.py` | Reads ZIPs from GDrive (rclone) or local |
| `pre_processamento/etl/transform.py` | Book reconstruction → 9 stationary features |
| `pre_processamento/etl/load.py` | Saves Parquet with compression |
| `pre_processamento/etl/validate.py` | NaN check, chronological order, gap detection |
| `pre_processamento/orchestration/run_pipeline.py` | ETL orchestrator |
| `pre_processamento/configs/cloud_config.yaml` | Paths, sampling rates, OB levels |
| `treino/run_training.py` | Full training loop: AdamW + CosineAnnealingLR + early stopping |
| `treino/training_config.yaml` | Final hyperparameters after Optuna |

### Auditor Model (`src/cloud/auditor_model/`)

| File | Purpose |
|:---|:---|
| `configs/auditor_config.yaml` | Walk-forward K=5, XGBoost params, base model HPs |
| `feature_engineering_meta.py` | 14 meta-features from base model output (no warm-up) |
| `train_xgboost.py` | OOF Walk-Forward K-Fold XGBoost training |
| `binance_adapter.py` | Binance Futures L2 WS + REST bootstrap (live inference) |

## Tests (`tests/`)

| File | Purpose |
|:---|:---|
| `test_cloud_etl_output.py` | Validates ETL output: 810 cols, no NaNs, OB sorting |
| `test_labelling_output.py` | Validates label distributions and data integrity |
| `test_preprocessed_quality.py` | Row count, schema, null checks on pre-processed files |
| `conftest.py` | Pytest fixtures |

## Docs (`docs/`)

| File | Purpose |
|:---|:---|
| `TCN_LSTM.md` | **Primary reference** — full architecture, constraints, execution |
| `1_SETUP_AND_ENV.md` | Hardware, CUDA, RunPod setup |
| `2_DATA_COLLECTION.md` | Bybit L2 download process |
| `3_DATA_ENGINEERING.md` | ETL feature engineering details |
| `5_LABELING_STRATEGY.md` | Labelling logic and thresholds |
| `7_OPERATIONAL_MANUAL.md` | RunPod execution guide |

## Configuration Files

| File | Purpose |
|:---|:---|
| `requirements.txt` | Python dependencies |
| `pytest.ini` | Test configuration |
| `.gitignore` | Excludes venv, data, models, logs |
| `rclone.conf` | GDrive mount config (RunPod) |
