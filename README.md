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
  pre_processed/*.parquet (833 colunas, ~1.440 linhas/dia)
        ↓
    Labelling             ← run_labelling.py: threshold assimétrico lookahead=60min
        ↓
  labelled_*/*.parquet (833 colunas + target ∈ {0,1,2})
        ↓
┌─────────────────────────────────────────────┐
│         BASE MODEL — Hybrid_TCN_LSTM        │
│  Input: (B, 720, 32) — 12h × 32 features    │
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
│   ├── L2/pre_processed/    ← Output do ETL (833 cols Parquet)
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
| ⚙️ **[3_DATA_ENGINEERING.md](docs/3_DATA_ENGINEERING.md)** | ETL: schema 833 cols, 32 features com fórmulas, normalização |
| 🏷️ **[4_LABELING_STRATEGY.md](docs/4_LABELING_STRATEGY.md)** | Thresholds assimétricos, como gerar novos |
| 🤖 **[5_MODEL_ARCHITECTURE.md](docs/5_MODEL_ARCHITECTURE.md)** | **Referência arquitetural** — TCN+LSTM, XGBoost, constraints, OOF, live adapter |
| 🚁 **[6_OPERATIONAL_MANUAL.md](docs/6_OPERATIONAL_MANUAL.md)** | Pipeline 6 passos, guia RunPod, troubleshooting |
| 📊 **[7_DATA_REFERENCE.md](docs/7_DATA_REFERENCE.md)** | Referência técnica detalhada: schema raw, 32 features, labelling, normalização |

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

### Treino Completo na Nuvem (RunPod - Pipeline Ponta a Ponta)

Este guia consolidado ensina o passo a passo definitivo para preparar um ambiente RunPod absoluto do zero e rodar individualmente os scripts do pipeline (ETL -> Labelling -> Treinamento Optuna / Fundação / Especialista).

Cole os comandos linha a linha no Web Terminal ou via SSH do seu RunPod recém-criado.

#### Passo 1: Preparação do Sistema Operacional (Ubuntu)

```bash
apt-get update && apt-get install -y nano tmux pciutils wget curl unzip zip htop sudo software-properties-common rsync
```

#### Passo 2: Instalação do Rclone e Clonagem do Repositório

Aqui configuraremos o ambiente focado num fluxo de dados 100% contido dentro da pasta raiz do projeto.

**2.1 - Instalar Rclone:**
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

**2.2 - Clonar o Código para dentro da Nuvem:**
```bash
cd /workspace
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git
cd QuantGod_Cloud_TCNLSTM
git checkout tcn_lstn_features
```

**2.3 - Configurar Token Rclone Local no Projeto:**
```bash
nano rclone.conf
```
Dentro do Nano, cole a sua chave do Drive:
```ini
[drive]
type = drive
scope = drive
token = {"access_token":"ya29..."} # Substitua pelo token completo
```
*(Salve: `Ctrl+O`, `Enter`, `Ctrl+X`).*

**2.4 - Ambiente Virtual:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest-xdist
```

---

#### Passo 3: O Botão do Pânico (Limpeza Mestra da VM)

Se você já rodou execuções antigas, estragou algo ou o path de arquivos pirou em gerações externas (`/workspace/data`), rode esta marreta para começar o sistema LIMPO de volta ao zero:

```bash
cd /workspace
rm -rf data logs                         # Destrói dados e logs gerados incorretamente fora do repositório
cd /workspace/QuantGod_Cloud_TCNLSTM
git reset --hard HEAD
git pull origin tcn_lstn_features
rm -rf data/L2/raw/* data/L2/pre_processed/* data/L2/labelled* logs/* models/* artifacts/* .pytest_cache
```

---

#### Passo 4: Iniciando Pipeline (Via Tmux)

O treinamento dura horas. Isole-o num terminal inquebrável por queda de internet.
```bash
tmux new -s quantgod
```
*(Para sair e deixar rodando: aperte `Ctrl+B`, solte, aperte `D`. Para voltar depois: `tmux attach -t quantgod`)*

No Tmux, **ative o ambiente e injete a raiz no Kernel do Python**:
```bash
cd /workspace/QuantGod_Cloud_TCNLSTM
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/workspace/QuantGod_Cloud_TCNLSTM"
```

---

#### Passo 5: Executar o Fluxo (Comandos Diretos)

##### ▶️ ETAPA 1: Pré-Processamento (ETL de 16 Atributos)
**Objetivo:** Baixar ZIPs puros cru e explodir a física quântica dos Ticks em Parquets agrupados por 1 Minuto.
```bash
python src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py
```
> **Verificação Imediata:** Verifique se as contas bateram e o dado ETL é válido:
```bash
pytest tests/etl/test_cloud_etl_output.py -v -n 12
```
> **Salvar na Nuvem (Google Drive):** Envie para o Drive ANTES de prosseguir:
```bash
rclone copy data/L2/pre_processed drive:PROJETOS/PRE_PROCESSED_L2_2023_2026_1_MINUTE_32_FEATURES/ --config rclone.conf -P
```

##### ▶️ ETAPA 2: Labelling (Rótulos do Futuro)
**Objetivo:** Projetar rentabilidade futura (+0.4% / -0.4%) e assentar Classes 0, 1 e 2 dinamicamente nas pastas.
```bash
python src/cloud/base_model/labelling/run_labelling.py
```
> **Verificação Imediata:** Verifique a validade global dos rótulos (limiares matemáticos de precisão):
```bash
pytest tests/labelling/test_labelling_output.py -v -n 12
```
> **Salvar na Nuvem (Google Drive):** Proteja os Rótulos no seu Drive:
```bash
rclone copy data/L2/labelled_* drive:PROJETOS/LABELLED_L2_2023_2026_1_MINUTE_32_FEATURES/ --config rclone.conf -P
```

##### ▶️ ETAPA 3: Divisão Cronológica (Anti-Leakage)
**Objetivo:** Fatiar ordenadamente no tempo os dados rotulados nas pastas `train`, `val` e `test`. Esta etapa é **obrigatória** para prevenir vazamento de dados (onde a rede decora o dataset na validação).
```bash
python src/cloud/base_model/treino/split_dataset.py
```
> **Nota de Segurança:** O script `experiment_utils.py` bloqueia tentativas de rodar o Optuna se esta pasta `splits_` não for detectada para garantir validação Out-of-Sample limpa.

##### ▶️ ETAPA 4: Treinamento Pesado (Finetuning/Fundação)
**Objetivo:** Rodar a busca Optuna, encontrar o Top 1, salvar o modelo campeão e gerar validações matemáticas.
**ATENÇÃO:** O arquivo `src/cloud/base_model/otimizacao/optimization_config.yaml` já está configurado com os caminhos como `"AUTO"`, o que significa que o sistema descobrirá a pasta `splits_...` automaticamente.
```bash
python src/cloud/base_model/otimizacao/run_optuna.py
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
