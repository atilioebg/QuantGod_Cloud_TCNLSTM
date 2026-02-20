# 📊 data/ — Data Directory

Centraliza o ciclo de vida completo dos dados do pipeline QuantGod Cloud,
desde os arquivos brutos até os artefatos prontos para inferência.

> **Git:** todos os arquivos de dados são ignorados pelo `.gitignore` (tamanho). Apenas este `README.md` e os `.gitkeep` são versionados. Para recriar a estrutura em um novo ambiente, os diretórios são recriados automaticamente pelos scripts do pipeline ao executar.

---

## 📂 Estrutura Completa

```
data/
├── L2/                                         ← 35.7 GB | 10.136 arquivos
│   ├── pre_processed/                          ← 1.126 arquivos | ~6.0 GB
│   ├── raw/                                    ← Vazio (ZIPs ficam no GDrive)
│   ├── labelled/                               ← Vazio (pasta de trabalho)
│   ├── labelled_SELL_0003_BUY_0005_1h/        ← 1.126 arquivos | ~3.8 GB
│   ├── labelled_SELL_0004_BUY_0004_1h/        ← 1.126 arquivos | ~3.8 GB
│   ├── labelled_SELL_0004_BUY_0004_2h/        ← 1.126 arquivos | ~3.6 GB
│   ├── labelled_SELL_0004_BUY_0005_1h/        ← 1.126 arquivos | ~3.8 GB
│   ├── labelled_SELL_0004_BUY_0006_1h/        ← 1.126 arquivos | ~3.8 GB
│   ├── labelled_SELL_0004_BUY_0008_1h/        ← 1.126 arquivos | ~3.8 GB ✅ ativo
│   ├── labelled_SELL_0004_BUY_0008_2h/        ← 1.126 arquivos | ~3.6 GB
│   └── labelled_SELL_0004_BUY_001_2h/         ← 1.126 arquivos | ~3.6 GB
├── models/                                     ← Vazio (aguarda treino no RunPod)
├── live/                                       ← Vazio (aguarda inferência ao vivo)
├── processed/                                  ← Vazio (reservado, não utilizado atualmente)
├── artifacts/
│   └── plots/                                  ← Vazio (gráficos de diagnóstico)
└── README.md                                   ← Este arquivo
```

---

## 📁 `L2/`

Dados de Level 2 (Orderbook Depth) do par BTCUSDT/Bybit Futures.
É a pasta principal do pipeline e ocupa ~35.7 GB no total.

### `L2/raw/`

**Status:** Vazio localmente — os arquivos fonte ficam no **Google Drive** e são acessados via
rclone mount (drive Z: no RunPod).

**Formato dos arquivos fonte no GDrive:**
```
YYYY-MM-DD_BTCUSDT_ob200.data.zip   ← 2024 em diante (OB200)
YYYY-MM-DD_BTCUSDT_ob500.data.zip   ← 2023 (OB500 — processado com hard cut de 200 níveis)
```

**Como acessar:**
```bash
# RunPod — montar drive
rclone mount drive: /workspace/gdrive --daemon

# Local (Windows)
rclone mount drive: Z: --config rclone.conf
```

---

### `L2/pre_processed/`

**Status:** 1.126 arquivos Parquet | ~6.0 GB

**Gerado por:** `src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py`

**Formato de arquivo:**
```
YYYY-MM-DD_BTCUSDT_ob500.data.parquet   (~4.1 MB/arquivo)
```

**Conteúdo de cada arquivo (1 dia = 1 arquivo):**
- ~1.440 linhas (1 linha por minuto — resultado do resample de 1 segundo → 1 minuto)
- Colunas de orderbook: `bid_{0..199}_p`, `bid_{0..199}_s`, `ask_{0..199}_p`, `ask_{0..199}_s` (800 colunas)
- 9 features estacionárias de microestrutura:

| Feature | Descrição |
|:---|:---|
| `body` | Log-retorno Open→Close dentro do candle |
| `upper_wick` | Sombra superior normalizada pelo close anterior |
| `lower_wick` | Sombra inferior normalizada pelo close anterior |
| `log_ret_close` | Log-retorno do Close em relação ao Close anterior |
| `volatility` | Desvio padrão dos micro-preços dentro do minuto |
| `max_spread` | Spread máximo bid-ask observado no minuto |
| `mean_obi` | Média do Order Book Imbalance L0 no minuto |
| `mean_deep_obi` | Média do Deep OBI (top 5 níveis) no minuto |
| `log_volume` | log1p(tick_count) — proxy de volume por número de ticks |

- Coluna `close` (micro-price no fechamento — usada para derivar log_ret_close do próximo candle)

**Validação:**
```bash
pytest tests/test_cloud_etl_output.py   # 810 colunas, OB sorted, sem NaNs, ordem cronológica
pytest tests/test_preprocessed_quality.py   # ~1440 linhas/dia, schema, monotonicidade
```

---

### `L2/labelled_*/`

Oito experimentos de labelling com configurações distintas de threshold e lookahead.
Cada pasta contém os **mesmos 1.126 arquivos Parquet** do `pre_processed/`, acrescidos da coluna `target`.

**Convenção de nomenclatura:**
```
labelled_SELL_{threshold_short}_BUY_{threshold_long}_{lookahead}
```

| Pasta | Short threshold | Long threshold | Lookahead | Tamanho |
|:---|:---:|:---:|:---:|:---:|
| `labelled_SELL_0003_BUY_0005_1h` | -0.3% | +0.5% | 60 min | ~3.8 GB |
| `labelled_SELL_0004_BUY_0004_1h` | -0.4% | +0.4% | 60 min | ~3.8 GB |
| `labelled_SELL_0004_BUY_0004_2h` | -0.4% | +0.4% | 120 min | ~3.6 GB |
| `labelled_SELL_0004_BUY_0005_1h` | -0.4% | +0.5% | 60 min | ~3.8 GB |
| `labelled_SELL_0004_BUY_0006_1h` | -0.4% | +0.6% | 60 min | ~3.8 GB |
| `labelled_SELL_0004_BUY_0008_1h` | -0.4% | +0.8% | 60 min | ~3.8 GB | **← ativo** |
| `labelled_SELL_0004_BUY_0008_2h` | -0.4% | +0.8% | 120 min | ~3.6 GB |
| `labelled_SELL_0004_BUY_001_2h` | -0.4% | +1.0% | 120 min | ~3.6 GB |

**Coluna `target`:**
| Valor | Classe | Condição |
|:---:|:---|:---|
| `0` | SELL | `future_return < threshold_short` |
| `1` | NEUTRAL | Retorno entre os thresholds |
| `2` | BUY | `future_return > threshold_long` |

> O experimento **`labelled_SELL_0004_BUY_0008_1h`** é o configurado como ativo em `labelling_config.yaml` e `base_model_config.yaml`.

**Validação:**
```bash
pytest tests/test_labelling_output.py
```

---

## 📁 `models/`

**Status:** Vazio — será populado após o treinamento no RunPod.

**Arquivos esperados após execução:**
```
models/
├── best_tcn_lstm.pt           ← Checkpoint do Hybrid_TCN_LSTM (melhor F1 Macro)
├── scaler_finetuning.pkl      ← StandardScaler (fit no train set) — usado em treino e live
└── xgb_auditor.json           ← Modelo XGBoost treinado (walk-forward OOF)
```

> ⚠️ O `scaler_finetuning.pkl` é **crítico** para inferência ao vivo. O `binance_adapter.py` e o `feature_engineering_meta.py` dependem dele para normalizar features com a mesma distribuição do treino.

---

## 📁 `live/`

**Status:** Vazio — será populado durante inferência ao vivo via `binance_adapter.py`.

**Arquivos esperados:**
```
live/
└── last_candles.parquet   ← Janela deslizante dos últimos 720 candles normalizados
```

---

## 📁 `processed/`

**Status:** Vazio — reservado para uso futuro.

> Mantida para compatibilidade com pipelines legados. Não utilizada no pipeline TCN+LSTM atual.

---

## 📁 `artifacts/plots/`

**Status:** Vazio — gráficos de diagnóstico gerados manualmente.

**Uso:** Histogramas de distribuição de features, séries temporais de micro_price, matrizes de correlação — gerados durante análise exploratória e verificação de qualidade.

---

## ⚙️ Boas Práticas

### Não versionar dados no git
O `.gitignore` exclui toda a pasta `data/` exceto `README.md` e `.gitkeep`. **Nunca** fazer `git add data/`.

### Fonte única de verdade para paths
Todos os caminhos de data são definidos nos arquivos de configuração:
- `src/cloud/base_model/pre_processamento/configs/cloud_config.yaml` → ETL
- `src/cloud/base_model/labelling/labelling_config.yaml` → Labelling
- `src/cloud/base_model/treino/training_config.yaml` → Training
- `src/cloud/auditor_model/configs/auditor_config.yaml` → XGBoost

### Backup antes de deletar experimentos de labelling
Cada pasta `labelled_*/` representa ~3.6–3.8 GB de dados já processados (≈ 1h de ETL no RunPod). Antes de excluir, confirme que o experimento não é mais necessário:
```bash
rclone copy data/L2/labelled_SELL_0004_BUY_0008_1h drive:QuantGod/backups/labelled/
```
