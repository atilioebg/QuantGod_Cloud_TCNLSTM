# QuantGod E2E Train Solution 🚀

Este documento descreve os passos realizados para o teste de ponta a ponta local, desde o mount dos dados até o treinamento do modelo.

## 📌 Guia de Execução

### 1. Montagem do Google Drive (Mount Z:)
Utilizamos o `rclone.exe` na raiz do projeto para montar os dados brutos.
```powershell
Start-Process -FilePath ".\rclone.exe" -ArgumentList "mount drive: Z: --config .\rclone.conf --vfs-cache-mode full" -WindowStyle Hidden
```

### 2. Pré-processamento (ETL)
Processamos o subset de teste `btcusdt_L2_2026_test` localizado no Drive.
- **Configuração**: `src/cloud/base_model/pre_processamento/configs/test_local.yaml`
- **Comando**:
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe -m src.cloud.base_model.pre_processamento.orchestration.run_pipeline src/cloud/base_model/pre_processamento/configs/test_local.yaml
```

### 3. Teste do Pré-processamento
Validamos a integridade e qualidade dos arquivos Parquet gerados.
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_cloud_etl_output.py
.\venv\Scripts\python.exe -m pytest tests/test_preprocessed_quality.py
```

### 4. Labelling (Rotulagem)
Aplicamos as regras: **Buy +0.1%**, **Sell -0.1%** e **Timeframe 2h** (120 min).
- **Configuração**: `src/cloud/base_model/labelling/labelling_config_e2e.yaml`
- **Comando**:
```powershell
.\venv\Scripts\python.exe src/cloud/base_model/labelling/run_labelling.py src/cloud/base_model/labelling/labelling_config_e2e.yaml
```

### 5. Teste do Labelling
Verificamos se as classes estão balanceadas e se a estrutura do target está correta. (Obs: Definir $env:LABELLED_DIR para o teste).
```powershell
$env:PYTHONPATH="."
$env:LABELLED_DIR="data/L2/labelled_SELL_0001_BUY_0001_2h"
.\venv\Scripts\python.exe -m pytest tests/test_labelling_output.py
```

### 6. Split do Dataset
Dividimos os dados em pastas `train` e `val` para o treinamento.
```powershell
.\venv\Scripts\python.exe src/cloud/base_model/treino/split_dataset.py data/L2/labelled_SELL_0001_BUY_0001_2h
```

### 7. Fine-tunning com Optuna
Executamos a busca de hiperparâmetros (1 trial como teste).
- **Configuração**: `src/cloud/base_model/otimizacao/optimization_config_e2e.yaml`
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe src/cloud/base_model/otimizacao/run_optuna.py src/cloud/base_model/otimizacao/optimization_config_e2e.yaml
```

### 8. Treinamento Final
Realizamos o treino do modelo com os melhores parâmetros escolhidos pelo Optuna (via `best_params.json`).
- **Configuração**: `src/cloud/base_model/treino/training_config_e2e.yaml`
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe src/cloud/base_model/treino/run_training.py src/cloud/base_model/treino/training_config_e2e.yaml
```

---
**Data do Teste**: 2026-02-20
**Status**: COMPLETO E TESTADO COM SUCESSO! ✅
