# QuantGod Cloud ⚡

> **Repositório:** [`atilioebg/QuantGod_Cloud_TCNLSTM`](https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM) | **Branch:** `main` | **Status:** 🟢 Production Ready
>
> Sistema de predição de direção de mercado para **BTC/USDT Perpetual Futures (Bybit/Binance)** usando um ensemble **TCN+LSTM (Base Model) + XGBoost (Auditor)**, treinado em dados Level 2 de Order Book históricos de 2023–2026.

---

## 🧠 O que é o QuantGod?

QuantGod é um sistema de ML de ponta a ponta para sinais de trading em alta frequência. Dado um histórico de 12 horas de microestrutura de mercado (720 snapshots de orderbook de 1 minuto), o sistema emite um dos três sinais:

| Sinal | Código | Interpretação |
|:---:|:---:|:---|
| **SELL** | `0` | Retorno < -0.4% nos próximos 60 min |
| **NEUTRAL** | `1` | Ausência de direção clara — não negociar |
| **BUY** | `2` | Retorno > +0.8% nos próximos 60 min |

---

## 🏗️ Arquitetura do Sistema

```
Bybit L2 ZIPs (GDrive, 2023–2026)
        ↓
    ETL Pipeline          ← transform.py: book reconstruction, 9 features, 1min resample
        ↓
  pre_processed/*.parquet (810 colunas, ~1.440 linhas/dia)
        ↓
    Labelling             ← run_labelling.py: threshold assimétrico lookahead=60min
        ↓
  labelled_*/*.parquet (810 colunas + target ∈ {0,1,2})
        ↓
┌─────────────────────────────────────────────┐
│         BASE MODEL — Hybrid_TCN_LSTM        │
│  Input: (B, 720, 9) — 12h × 9 features    │
│  TCN Stack (dilations [1,2,4,8]) + LSTM    │
│  Output: {logits: (B,3), probs: (B,3)}    │
└──────────────────┬──────────────────────────┘
                   ↓ probs + last_step_features
┌─────────────────────────────────────────────┐
│        AUDITOR MODEL — XGBoost              │
│  14 meta-features (probs, entropy,         │
│  candle features, RSI, EMA distances)      │
│  Output: calibrated signal + confidence    │
└─────────────────────────────────────────────┘
        ↓
   Live Inference (Binance Futures WS)
```

---

## 📂 Estrutura do Repositório

```
QuantGod_Cloud/
├── src/cloud/
│   ├── base_model/          ← ETL, Labelling, TCN+LSTM, Optuna, Training
│   └── auditor_model/       ← XGBoost, Feature Engineering Meta, Binance Live Adapter
├── data/
│   ├── L2/pre_processed/    ← Output do ETL (810 cols Parquet)
│   ├── L2/labelled_*/       ← Datasets rotulados (+ coluna target)
│   ├── models/              ← Checkpoints: .pt, .pkl, .json
│   └── live/                ← Buffer de candles ao vivo
├── tests/                   ← Suite de testes (unitários + integridade + qualidade de dados)
├── docs/                    ← Documentação técnica completa
└── logs/                    ← Logs de ETL, labelling, optuna, training
```

---

## 📚 Documentação

| Documento | Conteúdo |
|:---|:---|
| 🗺️ **[0_REPO_MAP.md](docs/0_REPO_MAP.md)** | Mapa completo do repositório — arquivos, configs, artefatos |
| 🛠️ **[1_SETUP_AND_ENV.md](docs/1_SETUP_AND_ENV.md)** | Hardware, instalação de dependências, rclone, checklist |
| 📡 **[2_DATA_COLLECTION.md](docs/2_DATA_COLLECTION.md)** | Dados brutos Bybit L2, GDrive, acesso live via Binance |
| ⚙️ **[3_DATA_ENGINEERING.md](docs/3_DATA_ENGINEERING.md)** | ETL: schema 810 cols, 9 features com fórmulas, normalização |
| 🏷️ **[4_LABELING_STRATEGY.md](docs/4_LABELING_STRATEGY.md)** | Thresholds assimétricos, 8 experimentos, como gerar novos |
| 🤖 **[5_MODEL_ARCHITECTURE.md](docs/5_MODEL_ARCHITECTURE.md)** | **Referência arquitetural** — TCN+LSTM, XGBoost, constraints, OOF, live adapter |
| 🚁 **[6_OPERATIONAL_MANUAL.md](docs/6_OPERATIONAL_MANUAL.md)** | Pipeline 6 passos, guia RunPod, troubleshooting |
| 📊 **[7_DATA_REFERENCE.md](docs/7_DATA_REFERENCE.md)** | Referência técnica detalhada: schema raw, 9 features, labelling, normalização |

Para a documentação do pipeline de infraestrutura cloud completa, consulte também:
- 📋 **[src/cloud/README.md](src/cloud/README.md)** — Guia operacional completo

> **Ordem de leitura sugerida:** `0_REPO_MAP` → `1_SETUP` → `2_DATA_COLLECTION` → `3_DATA_ENGINEERING` → `4_LABELING` → `5_MODEL_ARCHITECTURE` → `6_OPERATIONAL_MANUAL` → `7_DATA_REFERENCE` (apêndice)

---

## 🚀 Quick Start

### Ambiente Local (Windows — desenvolvimento/testes)

```powershell
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git
cd QuantGod_Cloud_TCNLSTM
python -m venv venv && venv\Scripts\Activate.ps1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Testes rápidos (sem GPU, sem dados) — < 30 segundos
python -m pytest tests/test_config_integrity.py tests/test_meta_features.py tests/test_model.py -v
```

### Treino Completo (RunPod — GPU)

```bash
# Ver guia completo em docs/7_OPERATIONAL_MANUAL.md
# Ou em src/cloud/README.md → "Guia RunPod"
```

---

## 🧪 Suite de Testes

```bash
# Unitários (sem dados, sem GPU)
pytest tests/test_model.py tests/test_meta_features.py tests/test_config_integrity.py -v

# Qualidade de dados (requer data/L2/ populado)
pytest tests/test_cloud_etl_output.py tests/test_preprocessed_quality.py -v
pytest tests/test_labelling_output.py -v

# Trocar experimento de labelling
pytest tests/test_labelling_output.py --labelled-dir data/L2/labelled_SELL_0004_BUY_0006_1h -v
```

Consulte **[tests/README.md](tests/README.md)** para documentação completa da suite.

---

## 🔑 Decisões de Design

| Decisão | Motivo |
|:---|:---|
| **TCN+LSTM** ao invés de Transformer puro | O ViViT colapsou em F1≈0.29 — apenas classe NEUTRAL. TCN garante causalidade local, LSTM mantém memória de longo prazo. |
| **XGBoost como Auditor** | Calibra e filtra predições do base model usando meta-features; treinado em OOF para zero leakage |
| **Thresholds assimétricos** (BUY=+0.8%, SELL=-0.4%) | Reflete assimetria real de risco/retorno em futuros de BTC |
| **seq_len=720** (12 horas) | Captura contexto de sessão de mercado sem aumentar VRAM exponencialmente |
| **StandardScaler fit apenas no train** | Garante zero leakage de distribuição entre treino e validação |
| **rclone mount** (não download) | Evita ocupar NVMe local com dados brutos; dados de 2023–2026 excedem capacidade local |
| **F1 Macro** como métrica principal | Evita que a classe dominante (NEUTRAL ~65%) mascare erros em SELL/BUY |

---

## 📋 Dependências Principais

| Biblioteca | Versão | Uso |
|:---|:---|:---|
| `torch` | ≥ 2.1 | `Hybrid_TCN_LSTM` |
| `xgboost` | ≥ 2.0 | Auditor model |
| `polars` | ≥ 0.19 | ETL + Labelling |
| `scikit-learn` | ≥ 1.3 | StandardScaler, TimeSeriesSplit |
| `optuna` | ≥ 3.4 | Hyperparameter search |
| `numpy` | ≥ 1.24 | Feature engineering |
| `pyyaml` | ≥ 6.0 | Carregamento de configs |

Veja `requirements.txt` para a lista completa.

---

## 🌿 Branches

| Branch | Status | Descrição |
|:---|:---|:---|
| `main` | 🟢 **Ativo** | Arquitetura atual — TCN+LSTM ensemble |
