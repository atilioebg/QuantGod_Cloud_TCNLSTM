# ☁️ QuantGod Cloud Infrastructure — `src/cloud/`

Pipeline modular de ETL, Labelling, Treinamento e Inferência do QuantGod, projetado para executar em instâncias de nuvem (RunPod) ou localmente.

---

## 📂 Estrutura de Diretórios

```
src/cloud/
├── README.md                                 ← Este arquivo
│
├── base_model/                               ← Modelo base: Hybrid TCN+LSTM
│   ├── configs/
│   │   └── master_config.yaml                ← ⭐ Single Source of Truth (SSOT)
│   ├── labelling/
│   │   └── run_labelling.py                  ← Gerador de targets SELL/NEUTRAL/BUY
│   ├── models/
│   │   └── model.py                          ← Hybrid_TCN_LSTM (CausalConv1D + LSTM + MLP)
│   ├── otimizacao/
│   │   ├── run_optuna.py                     ← Busca bayesiana de hiperparâmetros
│   │   └── best_params.json                  ← Resultado do Optuna (gerado automaticamente)
│   ├── pre_processamento/
│   │   ├── orchestration/run_pipeline.py     ← Orquestrador do ETL
│   │   ├── etl/extract.py                    ← Leitura de ZIPs (GDrive/local)
│   │   ├── etl/transform.py                  ← Reconstrução do book → 9 features
│   │   ├── etl/load.py                       ← Serialização Parquet
│   │   └── etl/validate.py                   ← NaN check, order, gaps
│   └── treino/
│       └── run_training.py                   ← Loop de treino final
│
└── auditor_model/                            ← Modelo auditor: XGBoost
    ├── feature_engineering_meta.py           ← 14 meta-features sem warm-up
    ├── train_xgboost.py                      ← XGBoost walk-forward trainer
    ├── auditor_labelling.py                  ← Dataset Fuser (Logits + Contexto)
    └── binance_adapter.py                    ← Integração live Binance Futures WS
```

---

## 🛠️ Configuração do Ambiente

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

### Conexão com Dados (rclone)

O pipeline não baixa os dados brutos para o disco local. Usa **stream via mount** do Google Drive.

**Windows (dev local):**
```powershell
.\rclone.exe mount drive: Z: --vfs-cache-mode full --config rclone.conf
# ⚠️ Mantenha esta janela aberta enquanto trabalhar
```

**Linux (RunPod):**
```bash
mkdir -p /workspace/gdrive
rclone mount drive: /workspace/gdrive --vfs-cache-mode full --allow-other &
```

> **Atenção ao path:** edite `master_config.yaml` conforme o ambiente:
> - Local Windows: `raw_l2_source: "drive:PROJETOS/..."`
> - RunPod/Linux: `raw_l2_source: "drive:PROJETOS/..."` (rclone cuida do mapeamento)

---

## 🚀 Pipeline de Execução — Ordem Obrigatória

```
[1. ETL] → [2. Labelling] → [3. Optuna] → [4. Training] → [5. XGBoost] → [6. Live]
```

---

## 1️⃣ ETL — Pré-Processamento

**Script:** `base_model/pre_processamento/orchestration/run_pipeline.py`
**Config:** `base_model/configs/master_config.yaml`

```bash
# RunPod (usar tmux para processos longos)
tmux new -s etl
export PYTHONPATH=$PYTHONPATH:/workspace
python -m src.cloud.base_model.pre_processamento.orchestration.run_pipeline
```

**O que faz:**
1. `extract.py` — Lista e lê ZIPs recursivamente do GDrive (suporta ob200 e ob500)
2. `transform.py` — Reconstrói o orderbook tick a tick → resample 1 segundo → 1 minuto → calcula 9 features estacionárias
3. `load.py` — Salva em `data/L2/pre_processed/YYYY-MM-DD_BTCUSDT_ob*.parquet`
4. `validate.py` — Verifica NaNs, Infinity, ordenação temporal e gaps

**Features geradas (9 colunas):**

| Feature | Descrição |
|:---|:---|
| `body` | Log-retorno Open→Close do candle |
| `upper_wick` | Sombra superior / close anterior |
| `lower_wick` | Sombra inferior / close anterior |
| `log_ret_close` | Log-retorno Close→Close |
| `volatility` | Std dos micro-preços no minuto |
| `max_spread` | Spread bid-ask máximo do minuto |
| `mean_obi` | Média do Order Book Imbalance L0 |
| `mean_deep_obi` | Média do Deep OBI (top 5 níveis) |
| `log_volume` | log1p(tick_count) — proxy de volume |

**Validação:**
```bash
pytest tests/test_cloud_etl_output.py
pytest tests/test_preprocessed_quality.py
```

**Hardware recomendado:** 8+ vCPUs / 16 GB RAM (CPU-bound, sem GPU).

---

## 2️⃣ Labelling — Geração de Targets

**Script:** `base_model/labelling/run_labelling.py`
**Config:** `base_model/configs/master_config.yaml`

```bash
python -m src.cloud.base_model.labelling.run_labelling
```

**Lógica de Labelling (Thresholds Assimétricos):**

| Label | Valor | Condição |
|:---:|:---|:---|
| `0` | SELL | `future_return < threshold_short` (default: -0.4%) |
| `1` | NEUTRAL | Entre os thresholds |
| `2` | BUY | `future_return > threshold_long` (default: +0.8%) |

O retorno futuro é calculado sobre uma janela de `lookahead` minutos (default: 60 min).

**Experimentos disponíveis em `data/L2/`:**

| Pasta | Short | Long | Lookahead |
|:---|:---:|:---:|:---:|
| `labelled_SELL_0004_BUY_0008_1h` | -0.4% | +0.8% | 60 min | **← ativo** |
| `labelled_SELL_0004_BUY_0006_1h` | -0.4% | +0.6% | 60 min | |
| `labelled_SELL_0004_BUY_0005_1h` | -0.4% | +0.5% | 60 min | |
| `labelled_SELL_0004_BUY_0004_1h` | -0.4% | +0.4% | 60 min | |
| `labelled_SELL_0004_BUY_0008_2h` | -0.4% | +0.8% | 120 min | |
| `labelled_SELL_0004_BUY_0004_2h` | -0.4% | +0.4% | 120 min | |
| `labelled_SELL_0004_BUY_001_2h` | -0.4% | +1.0% | 120 min | |
| `labelled_SELL_0003_BUY_0005_1h` | -0.3% | +0.5% | 60 min | |

**Validação:**
```bash
pytest tests/test_labelling_output.py
```

**Output:** `data/L2/labelled_{CONFIG}/YYYY-MM-DD_BTCUSDT_ob*.parquet` — mesmo schema do `pre_processed/` + coluna `target`.

---

## 3️⃣ Optuna — Otimização de Hiperparâmetros

**Script:** `base_model/otimizacao/run_optuna.py`
**Config:** `base_model/configs/master_config.yaml`

```bash
python -m src.cloud.base_model.otimizacao.run_optuna
```

**Configuração:**

| Parâmetro | Valor |
|:---|:---|
| Trials | 30 |
| Timeout | 10 horas |
| Métrica | F1 Macro (não weighted — evita dominância do NEUTRAL) |
| Pruner | `MedianPruner(n_startup_trials=5, n_warmup_steps=2)` |

**Search Space:**

| Hiperparâmetro | Valores Candidatos |
|:---|:---|
| `tcn_channels` | 32, 64, 128 |
| `lstm_hidden` | 128, 256, 512 |
| `num_lstm_layers` | 1, 2 |
| `lr` | [5e-5, 1e-3] (log-uniform) |
| `dropout` | [0.2, 0.5] |
| `batch_size` | 128, 256 (512 excluído — OOM com lstm=512 + seq=1440) |
| `seq_len` | 720 (12h), 1440 (24h) |
| `epochs` por trial | 5 |

**OOM Guard:** Se qualquer trial levantar `RuntimeError("out of memory")`, o código executa `torch.cuda.empty_cache()` e registra `optuna.TrialPruned()` — nunca trava a busca.

**Outputs:**
- `best_params.json` — parâmetros do melhor trial
- `optuna_tcn_lstm_v1.db` — histórico completo (SQLite)

```bash
# Visualizar dashboard Optuna
optuna-dashboard sqlite:///optuna_tcn_lstm_v1.db
# → http://127.0.0.1:8080
```

---

## 4️⃣ Treino Final — Base Model (`Hybrid_TCN_LSTM`)

**Script:** `base_model/treino/run_training.py`
**Config:** `base_model/configs/master_config.yaml` ← fonte única de verdade

```bash
python -m src.cloud.base_model.treino.run_training
```

### Arquitetura: `Hybrid_TCN_LSTM`

```
Input (B, 720, 9)
  → TCN Stack: 4 × CausalConv1D
      dilation=[1, 2, 4, 8], kernel=3
      + Residual connection + BatchNorm + GELU + SpatialDropout
  → LSTM (hidden=256, layers=2, batch_first=True, dropout=0.2)
      → last hidden state h_n[-1]: (B, 256)
  → MLP Head: Linear(256→128) → GELU → Dropout(0.4) → Linear(128→3)
Output: { "logits": (B, 3), "probs": softmax(B, 3) }
```

**Causal Convolution:** padding = `dilation × (kernel_size − 1)` à esquerda + trim à direita. `output[t]` depende apenas de `input[t-k], k≥0`. Zero leakage de futuro.

### Regime de Treinamento

| Parâmetro | Valor | Justificativa |
|:---|:---|:---|
| `seq_len` | 720 passos (12h) | Lookback que captura ciclos intraday completos |
| `batch_size` | 256 | Balanço I/O / VRAM |
| `lr` inicial | 0.0003 | Ponto de partida AdamW estável para LSTM |
| `optimizer` | `AdamW(weight_decay=0.01)` | L2 correto para sequências (vs Adam puro) |
| `scheduler` | `CosineAnnealingLR` | Decaimento suave, evita steps abruptos |
| `gradient_clip_norm` | 1.0 | Previne exploding gradients no LSTM BPTT |
| `loss` | `CrossEntropyLoss(weight=[2.0, 1.0, 2.0])` | SELL e BUY 2× mais penalizados que NEUTRAL |
| `early_stopping` | patience=3, critério=F1 Macro | Para no melhor F1 real, não no menor loss |
| `epochs` máx | 10 | |

### Divisão do Dataset

```
Dataset cronológico completo (1.126 dias, 2023-01 a 2026-02):

├── Treino (80% primeiros dias)
│   └── StandardScaler fit APENAS aqui
└── Validação (20% últimos dias — nunca visto no treino)
    └── Critério de early stopping (F1 Macro)
```

> ⚠️ **Zero look-ahead:** o split é estritamente temporal. O scaler nunca vê dados de validação. O `StandardScaler` fitted é salvo em `data/models/scaler_finetuning.pkl` — usado idêntico em treino, validação e inferência ao vivo.

### Outputs

| Arquivo | Conteúdo |
|:---|:---|
| `data/models/best_tcn_lstm.pt` | Checkpoint com melhor F1 Macro na validação |
| `data/models/scaler_finetuning.pkl` | StandardScaler fitted no train set |

---

## 5️⃣ XGBoost Auditor — Walk-Forward OOF

**Script:** `auditor_model/train_xgboost.py`
**Eng. Features:** `auditor_model/feature_engineering_meta.py`
**Config:** `base_model/configs/master_config.yaml`

```bash
python -m src.cloud.auditor_model.train_xgboost
```

### Regime de Treinamento: Out-of-Fold Walk-Forward

O XGBoost é um **segundo estágio** que aprende a calibrar, corrigir e vetar as predições do TCN+LSTM. Para evitar qualquer leakage — o XGBoost **nunca** vê predições geradas em dados que o TCN+LSTM usou para treinar.

**Protocolo de divisão:**

```
Dataset completo (cronológico)
├── 90% — DEV set
│   Dividido em K=5 folds via TimeSeriesSplit (temporal, sem shuffle)
│   │
│   ├── Fold 1: [TRAIN → TCN+LSTM] | Fold 2: [OOF predictions geradas]
│   ├── Fold 2: [TRAIN → TCN+LSTM] | Fold 3: [OOF predictions geradas]
│   ├── Fold 3: [TRAIN → TCN+LSTM] | Fold 4: [OOF predictions geradas]
│   └── Fold 4: [TRAIN → TCN+LSTM] | Fold 5: [OOF predictions geradas]
│
│   Pool OOF acumulado = OOF₂ + OOF₃ + OOF₄ + OOF₅
│   XGBoost TREINA neste pool
│
└── 10% — TEST set (nunca tocado durante nenhuma etapa)
    └── Avaliação final do ensemble (TCN+LSTM + XGBoost)
```

> **Regra inviolável:** O XGBoost treina EXCLUSIVAMENTE em predições OOF — nunca em predições in-sample.

### 14 Meta-Features (Input do XGBoost)

| # | Feature | Fonte |
|:---|:---|:---|
| 0–2 | `prob_sell, prob_neutral, prob_buy` | Output `softmax` do TCN+LSTM |
| 3 | `entropy` = -Σ p·log(p) | Incerteza do modelo |
| 4–10 | `body, upper_wick, lower_wick, log_ret_close, volatility, mean_obi, mean_deep_obi` | Último timestep do tensor (t=720) |
| 11 | `rsi_14` | Calculado do tensor micro_price (sem warm-up) |
| 12 | `ema_9_dist` | % distância para EMA de 9 períodos |
| 13 | `ema_50_dist` | % distância para EMA de 50 períodos |

**Zero warm-up:** RSI e EMAs são extraídos da janela de 720 passos já em memória:
```python
log_rets = X_sequences[:, :, 3]                     # col log_ret_close
micro_prices = np.exp(np.cumsum(log_rets, axis=1))  # (B, 720)
rsi_14 = _rsi(micro_prices[j], period=14)           # 720 pontos disponíveis
```

**Excluídos:** `max_spread` (≥0.8 correlação com `volatility`), `log_volume` (baixo sinal pós-probs).

### Hiperparâmetros XGBoost

| Parâmetro | Valor |
|:---|:---|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| Objetivo | `multi:softprob` (3 classes) |
| `early_stopping_rounds` | 30 (avaliado no TEST set) |

### Outputs

| Arquivo | Conteúdo |
|:---|:---|
| `data/models/xgb_auditor.json` | Modelo XGBoost serializado |
| `data/models/scaler_auditor.pkl` | Scaler dos meta-features |

---

## 6️⃣ Binance Adapter — Inferência Live

**Script:** `auditor_model/binance_adapter.py`

```bash
# Modo produção
python src/cloud/auditor_model/binance_adapter.py

# Modo teste (2 candles e encerra)
python src/cloud/auditor_model/binance_adapter.py --test-mode --max-candles 2
```

**Fluxo de dados em produção:**
```
Binance Futures WebSocket (btcusdt@depth@100ms)
  ↓ Protocolo de sync: lastUpdateId / U / u
Orderbook local reconstruído (mesmo algo do ETL)
  ↓ resample 1 min → 9 features → StandardScaler
Buffer circular (720 candles)
  ↓ Tensor (1, 720, 9)
TCN+LSTM → probs (1, 3)
  ↓ + 14 meta-features (sem warm-up)
XGBoost → sinal calibrado (SELL / NEUTRAL / BUY)
```

**Protocolo de sincronização (Constraint crítico):**
1. Abre WebSocket, bufferiza mensagens (não aplica ainda)
2. Faz REST snapshot → obtém `lastUpdateId`
3. Drena buffer:
   - Descarta se `u ≤ lastUpdateId` (já no snapshot)
   - Re-bootstrap se `U > lastUpdateId + 1` (gap)
   - Aplica se `U ≤ lastUpdateId + 1 ≤ u` (continuidade válida)
4. Aplica deltas live normalmente

**Detecção de drift:** cada feature é checada contra o Z-score do scaler de treino. Se `|z| > 4.0` → `WARNING` logado.

---

## 📂 Logs

Todos os logs vão para `logs/` na raiz do projeto. Consulte [`logs/LOGS_README.md`](../../logs/LOGS_README.md) para detalhes completos.

| Pasta | Script | O que registra |
|:---|:---|:---|
| `logs/etl/` | `run_pipeline.py` | Arquivos processados, NaNs, gaps |
| `logs/labelling/` | `run_labelling.py` | Distribuição SELL/NEU/BUY por arquivo |
| `logs/optimization/` | `run_optuna.py` | F1 por trial, OOM pruned, best params |
| `logs/training/` | `run_training.py` | Loss/F1 por epoch, checkpoint salvo |

---

## ⚡ Hardware Recomendado

| Etapa | CPU | RAM | GPU |
|:---|:---:|:---:|:---:|
| ETL | 8+ vCPUs | 16 GB | ❌ |
| Labelling | 4+ vCPUs | 8 GB | ❌ |
| Optuna | 4 vCPUs | 16 GB | ✅ RTX 3090+ |
| Training | 4 vCPUs | 16 GB | ✅ RTX 3090+ |
| XGBoost | 8+ vCPUs | 32 GB | ❌ |
| Live | 2 vCPUs | 4 GB | Opcional |

---

## 🆘 Troubleshooting

| Erro | Causa | Solução |
|:---|:---|:---|
| `path not found` ou `Z:\` vazio | rclone não montado | Rodar mount antes do pipeline |
| `CUDA out of memory` | batch_size / seq_len alto | Reduzir `batch_size` → 128; `seq_len` → 720 (já protegido no Optuna) |
| `U > lastUpdateId + 1` no adapter | Gap no WebSocket | Re-bootstrap automático (já implementado) |
| Labels com 0% SELL ou BUY | Threshold muito agressivo | Ajustar `labelling_config.yaml` e re-rotular |
| ob500 vs ob200 mismatch | Arquivo 2023 com 500 níveis | `transform.py` aplica hard cut automático para 200 níveis |

---

## 🔁 Guia Completo RunPod (Instância Zerada)

```bash
# 1. Clonar e configurar
cd /workspace
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git .
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:/workspace

# 2. Configurar rclone
mkdir -p /root/.config/rclone/
cp /workspace/rclone.conf /root/.config/rclone/rclone.conf
rclone lsd drive:   # validar conexão

# 3. Baixar dataset rotulado (instância GPU — baixar direto no NVMe)
tmux new -s download
mkdir -p /workspace/data/L2/labelled_SELL_0004_BUY_0008_1h
rclone copy drive:PROJETOS/L2/labelled_SELL_0004_BUY_0008_1h \
  /workspace/data/L2/labelled_SELL_0004_BUY_0008_1h -P
# Ctrl+B, D (detach)

# 4. Após download: Optuna
tmux new -s optuna
source venv/bin/activate && export PYTHONPATH=$PYTHONPATH:/workspace
python -m src.cloud.base_model.otimizacao.run_optuna

# 5. Treino final
python -m src.cloud.base_model.treino.run_training

# 6. XGBoost auditor
python -m src.cloud.auditor_model.train_xgboost

# 7. Backup dos modelos antes de desligar o Pod
rclone copy /workspace/data/models drive:PROJETOS/models -P
rclone copy /workspace/logs drive:PROJETOS/logs_backup -P
```
