# ⚙️ 3. Data Engineering (ETL)

> **Target Audience**: Data Engineers, Quants.
> **Script:** `src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py`
> **Config:** `src/cloud/base_model/pre_processamento/configs/cloud_config.yaml`

---

## 🚀 Execução

```bash
# Montar dados (rclone) antes de rodar
python -m src.cloud.base_model.pre_processamento.orchestration.run_pipeline

# Validar output
pytest tests/test_cloud_etl_output.py tests/test_preprocessed_quality.py -v
```

---

## 🔄 Fluxo do ETL

```
ZIP (GDrive/Bybit L2)
  │
  ├─ extract.py: Leitura recursiva de ZIPs (suporta ob200 e ob500)
  │
  ├─ transform.py: Reconstrução do orderbook tick a tick
  │   ├─ Aplicação de snapshots e deltas (size=0 → remoção de nível)
  │   ├─ Hard Cut: top 200 bids (desc) + top 200 asks (asc)
  │   ├─ Sampling: 1 tick/segundo (1000ms)
  │   ├─ Cálculo por tick: micro_price, spread, obi_l0, deep_obi_5, tick_count
  │   └─ Resample 1 minuto: OHLC + 9 features agregadas
  │
  ├─ load.py: Serialização em Parquet com compressão snappy
  │
  └─ validate.py: NaN check, Infinity, ordem temporal, gaps
```

---

## 📐 Hard Cut — ob500 → ob200

Arquivos de 2023 possuem 500 níveis de profundidade. O pipeline aplica corte automático:

```python
sorted_bids = sorted(self.bids_book.keys(), reverse=True)[:200]  # Top 200 bids
sorted_asks = sorted(self.asks_book.keys())[:200]                 # Top 200 asks
```

Isso garante **schema de colunas idêntico** para todos os anos (810 colunas), sem qualquer branch especial no código de treinamento.

---

## 📊 Output: Schema do Parquet

Cada arquivo `data/L2/pre_processed/YYYY-MM-DD_BTCUSDT_ob*.parquet` possui **810 colunas** e ~1.440 linhas (1 minuto por linha):

| Grupo | Padrão | Qtd | Descrição |
|:---|:---|:---:|:---|
| Bids — Preço | `bid_{0..199}_p` | 200 | Preço do nível i (bid_0 = best bid) |
| Bids — Tamanho | `bid_{0..199}_s` | 200 | Quantidade do nível i |
| Asks — Preço | `ask_{0..199}_p` | 200 | Preço do nível i (ask_0 = best ask) |
| Asks — Tamanho | `ask_{0..199}_s` | 200 | Quantidade do nível i |
| Features | *(ver abaixo)* | 9 | Input direto do modelo |
| Referência | `close` | 1 | Micro-price de fechamento |
| **TOTAL** | | **810** | |

**Ordenação garantida:** `bid_0_p > bid_1_p > ... > bid_199_p` (decrescente), `ask_0_p < ask_1_p < ... < ask_199_p` (crescente).

**Index:** `datetime` em UTC, frequência de 1 minuto.

---

## 🧮 As 9 Features Derivadas

Estas são as **únicas colunas** passadas como input ao modelo `Hybrid_TCN_LSTM`. Calculadas durante o resample de 1 minuto sobre os ticks de 1 segundo:

### Features de Candle (Forma da Vela)

| Feature | Fórmula | Descrição |
|:---|:---|:---|
| `body` | `log(close / open)` | Retorno log do corpo — positivo = alta, negativo = queda |
| `upper_wick` | `(high - max(open, close)) / prev_close` | Sombra superior normalizada pelo fechamento anterior |
| `lower_wick` | `(min(open, close) - low) / prev_close` | Sombra inferior normalizada pelo fechamento anterior |

> **OHLC** é derivado da `micro_price` durante o resample. `close` = micro_price no fechamento do minuto.

### Feature de Retorno

| Feature | Fórmula | Descrição |
|:---|:---|:---|
| `log_ret_close` | `log(close / prev_close)` | Log-retorno — série estacionária para ML |

> Esta coluna também é a base para reconstruir micro_price durante feature engineering do XGBoost (via `cumsum` dos log-retornos).

### Features de Microestrutura (Orderbook)

| Feature | Fórmula | Descrição |
|:---|:---|:---|
| `volatility` | `std(micro_price_ticks_1s)` | Desvio padrão da micro_price intra-candle |
| `max_spread` | `max(ask_0_p - bid_0_p)_ticks_1s` | Spread máximo bid-ask no minuto — proxy de stress de liquidez |
| `mean_obi` | `mean((bid_0_s - ask_0_s)/(bid_0_s + ask_0_s))` | OBI top 1 nível — range [-1, +1] |
| `mean_deep_obi` | `mean((Σbid_0..4_s - Σask_0..4_s)/(Σbid + Σask))` | OBI dos top 5 níveis — liquidez mais representativa |
| `log_volume` | `log1p(tick_count)` | Proxy de volume — count de mensagens L2 no minuto |

---

## 🔬 Variáveis Auxiliares (não são input do modelo)

| Variável | Descrição | Uso |
|:---|:---|:---|
| `micro_price` | `(bid_0_p × ask_0_s + ask_0_p × bid_0_s) / (bid_0_s + ask_0_s)` | Gera OHLC e log_ret_close |
| `spread` | `ask_0_p - bid_0_p` | Gera `max_spread` |
| `obi_l0` | `(bid_0_s - ask_0_s) / (bid_0_s + ask_0_s)` | Gera `mean_obi` |
| `deep_obi_5` | OBI dos top 5 níveis | Gera `mean_deep_obi` |
| `tick_count` | Count de mensagens L2/minuto | Gera `log_volume` |
| `close` | Micro-price de fechamento | Referência para labelling |
| `bid_{i}_p/s`, `ask_{i}_p/s` | Estado do book no fechamento do minuto | Referência para inspeção/debug |

---

## 📏 Normalização dos Inputs

Antes de entrar no modelo, as 9 features recebem Z-Score via `StandardScaler`:

```python
scaler = StandardScaler()
scaler.fit(X_train)      # Fit APENAS no conjunto de treino — sem leakage
X_train_norm = scaler.transform(X_train)
X_val_norm   = scaler.transform(X_val)
```

O scaler treinado é salvo em `data/models/scaler_finetuning.pkl` e carregado durante:
- Treino do XGBoost (normalização dos meta-features)
- Inferência live no `binance_adapter.py`

---

## 🕐 Timeframe dos Dados

```
Bybit WebSocket L2 ticks (~100ms de frequência)
    → Sampling 1s    → ~86.400 linhas/dia (1 por segundo)
    → Resample 1min  → ~1.440 linhas/dia  (1 por minuto) ← Output final
```

**Janela de lookback do modelo:** 720 candles × 1 min = **12 horas de histórico**.

---

## 🔍 Validação do Output ETL

```bash
# Verifica 810 colunas, ordenação do OB, sem NaN nas features, timestamps cronológicos
pytest tests/test_cloud_etl_output.py -v

# Verifica contagem de linhas (~1440/dia), continuidade de datas, schema
pytest tests/test_preprocessed_quality.py -v
```

Consulte [`data/README.md`](../data/README.md) para detalhes do volume de dados atual.
