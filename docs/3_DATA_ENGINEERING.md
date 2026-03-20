# ⚙️ 3. Data Engineering (ETL v5.0 - Information Domain)

> **Target Audience**: Data Engineers, Quants.
> **Script:** `src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py`
> **Config:** `src/cloud/base_model/configs/master_config.yaml` (Secção `pre_processing.etl`)

---

## 🚀 Filosofia: Do Tempo Cronológico para o Domínio da Informação

A versão 5.0 do QuantGod abandona a amostragem fixa de 1 minuto (Time-Bars). O mercado de alta frequência não opera em "minutos", mas em **eventos**. O novo ETL reconstrói o fluxo de dados em barras baseadas em métricas de atividade, garantindo que o modelo TCN-LSTM receba dados com variabilidade constante (IID - Independent and Identically Distributed).

---

## 🔄 Fluxo do ETL (Unified Event-Driven)

```
ZIP (Bybit L2) + CSV.GZ (Trades)
  │
  ├─ extract.py: Streaming de ZIPs (ob200/ob500) via Polars
  │
  ├─ merge_l2_trades.py: Join Atomic (join_asof) Backward
  │   └─ Alinha o estado do livro (L2) a cada Trade individual
  │
  ├─ event_sampler.py: Orquestração de Barras IID
  │   ├─ Tick Bars: Fecha barra a cada N trades (ex: 1000)
  │   ├─ Dollar Bars: Fecha barra a cada $N milhões (ex: 100k USD)
  │   ├─ Information Bars: Baseado em desequilíbrio de fluxo (OFI/VPIN)
  │   └─ Time Bars: Fallback cronológico (heartbeat de 5 min)
  │
  ├─ transform.py (Facade): Cálculo de micro_price, spread, OFI, Slope, RDI
  │
  ├─ load.py: Serialização em Parquet (Snappy) em `data/L2/pre_processed/`
  │
  │
  └─ validate.py: Auditoria de integridade (Island Split, Gap Detection)

---

## 🛡️ Resiliência e Escala: Scale Guard
O processamento de anos recentes (2025-2026) apresenta densidade de dados até 3x superior a 2023. Para evitar falhas de Out-of-Memory (OOM) em sistemas com 256GB de RAM:
- **Redutor Dinâmico**: O pipeline aplica um multiplicador (`scale_guard.thresholds`) sobre o total de CPU.
- **Memória por Worker**: Estimados **43GB/worker** para 2026.
- **Garbage Collection**: Forçado a cada 5 arquivos para liberar heaps do Polars/NumPy.
```

---

## 📐 Amostragem: O Conceito do "Transbordo"

Ao contrário de um simples `resample`, o `EventSampler` utiliza a técnica de **Reconciliação de Transbordo (Overflow)**. 
Se uma barra atinge o limite de 100k USD no meio de um trade de 150k USD:
1.  A barra atual fecha com os primeiros 100k.
2.  Os 50k excedentes são levados (carry forward) para a próxima barra.
Isso garante a **Conservação de Massa (Volume/Ticks)** total do dia e elimina o bias de arredondamento.

---

## 📊 Output: Schema do Parquet

Os arquivos agora possuem um número variável de linhas por dia (dependendo da volatilidade), mas mantêm a estrutura de **833 colunas**:

| Grupo | Padrão | Qtd | Descrição |
|:---|:---|:---:|:---|
| Bids — Preço | `bid_{0..199}_p` | 200 | Preço do nível i (bid_0 = best bid) |
| Bids — Tamanho | `bid_{0..199}_s` | 200 | Quantidade do nível i |
| Asks — Preço | `ask_{0..199}_p` | 200 | Preço do nível i (ask_0 = best ask) |
| Asks — Tamanho | `ask_{0..199}_s` | 200 | Quantidade do nível i |
| Features | *(ver 7_DATA_REFERENCE)* | 32 | Input direto do modelo |
| Identificador | `island_id` | 1 | ID da ilha (contiguidade temporal) |
| **TOTAL** | | **834** | |

---

## 🧮 As 32 Features Snipers

As features agregadas em cada barra representam o estado do mercado **durante aquele evento específico**:
1. **Core Shape:** `body`, `upper_wick`, `lower_wick`, `log_ret_close`, `volatility`.
2. **Micro-Flow:** `mean_obi`, `mean_deep_obi`, `ofi_l1_agg`, `buy_pressure_ratio`.
3. **Institutional:** `bid_slope`, `ask_slope`, `rdi_l1` (Relative Depth Imbalance).

---

## 🛡️ Validação e Auditoria (Island Split)

Devido aos gaps inerentes a dados brutos de exchange, o ETL implementa o **Island Split**:
*   Se o tempo entre dois trades excede 5 minutos (configurável), o pipeline interrompe a continuidade.
*   É gerado um novo `island_id`.
*   O modelo entende que as barras de ilhas diferentes não possuem relação sequencial, evitando "pulos" de indicadores.

---

## 🕐 Timeframe Adaptativo

```
Market Activity (Muito Lento) → Fecha por Time Barrier (ex: 5min)
Market Activity (Baleia)      → Fecha por Dollar Threshold
Market Activity (Frenético)   → Fecha por Tick Threshold
```

A TCN-LSTM enxerga todos esses objetos como passos equidistantes na série temporal, focando na **Informação** e não no relógio.

---

## 🔬 Modo de Amostragem Híbrido (CUSUM + Trigger)

O `sampling_mode` em `master_config.yaml` controla a estratégia de formação de barras:

| Modo | Lógica | Quando usar |
|:---|:---|:---|
| `trigger` | Trigger Bars puras (Dollar/Tick/OFI/Time) | Backtests rápidos, mercados trending |
| `cusum` | Somente Filtro CUSUM Simétrico de De Prado | Análise de quebras de regime |
| `both` | **CUSUM define janelas + Trigger constrói barras IID** | **Produção: TCN-LSTM** |

O filtro CUSUM detecta **desvios direcionais cumulativos** no preço. Ele acumula os run-ups (`sPos`) e run-downs (`sNeg`) e dispara um evento apenas quando o movimento é netamente direcional. 

### 📈 CUSUM Adaptativo (AFML v9.5)
O limiar $h$ (threshold de disparo) agora é **adaptativo** (`adaptive_h: true`):
*   $h[i] = \text{factor} \times \sigma_{\text{EWMA}}[i]$
*   A volatilidade é calculada tick-a-tick, permitindo que o filtro se ajuste automaticamente a mudanças de regime de mercado, evitando ruído em baixa vol e capturando explosões de momentum.

### 🚀 Upgrade de Performance: Numba

O loop CUSUM (`_cusum_loop` em `event_sampler.py`) é **path-dependent** (cada `sPos/sNeg` depende do tick anterior), portanto não paralelizável pelo Polars. Foi escrito em **NumPy puro** com estrita compatibilidade `@njit`.

Para ativar o Numba (ganho de ~50-100x em datasets grandes):

```bash
pip install numba
```

```python
# Em: src/cloud/base_model/pre_processamento/etl/event_sampler.py
# Adicionar no topo do arquivo:
from numba import njit

# Adicionar o decorator diretamente acima da função:
@njit  # ← ÚNICA linha de mudança necessária
def _cusum_loop(prices: np.ndarray, h_arr: np.ndarray) -> np.ndarray:
    ...
```

> **Upgrade Crítico (2026)**: O pipeline agora suporta **Distributed Processing** via `island_id`. Se a memória estourar, o Scale Guard reduz automaticamente o paralelismo, priorizando a estabilidade sobre a velocidade bruta.

---

Consulte [`3_ETL_GOLD_STANDARD.md`](3_ETL_GOLD_STANDARD_v5.0.md) para os critérios de aprovação de qualidade.
