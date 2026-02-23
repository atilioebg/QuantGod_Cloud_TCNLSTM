# Guia de Execução na Nuvem (RunPod) - Pipeline TCN-LSTM Ponta a Ponta

Este guia fornece o passo a passo definitivo para preparar um ambiente RunPod do absoluto zero e rodar individualmente os 3 scripts que compõem o pipeline final (ETL de 16 variáveis -> Labelling -> Treinamento Optuna / Fundação).

Cole os comandos linha a linha no Web Terminal ou via SSH do seu RunPod recém-criado.

---

## Passo 1: Preparação do Sistema Operacional e Ferramentas Básicas (Ubuntu)
Geralmente os templates do RunPod vêm em modo root e podem estar sem alguns pacotes vitais para edição de texto e gerência de sessões.

```bash
apt-get update && apt-get install -y nano tmux pciutils wget curl unzip zip htop sudo software-properties-common rsync
```

## Passo 2: Instalação do Rclone e Montagem do Google Drive

**2.1 - Baixar e instalar a versão oficial do Rclone para Linux via script:**
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

**2.2 - Criar o diretório raiz para o Workspace:**
(Aqui viverão os dados persistentes no volume de Network).
```bash
cd /workspace
mkdir -p data logs
```

**2.3 - Injetar o Token do Google Drive:**
```bash
nano /workspace/rclone.conf
```
Dentro do Nano, cole a configuração do seu token:
```ini
[drive]
type = drive
scope = drive
token = {"access_token":"ya29..."} # Substitua pela sua linha completa do token
```
*Salve e feche (`Ctrl+O`, `Enter`, `Ctrl+X`).*

## Passo 3: Clonagem do Repositório e Ambiente Virtual Python
Garanta que sua pasta raiz será `/workspace/`. O código fonte, o virtual env e os logs devem viver nela.

**3.1 - Clonar repositório e selecionar a branch correta:**
```bash
cd /workspace
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git
cd QuantGod_Cloud_TCNLSTM
git checkout tcn_lstn_features
```

**3.2 - Criar ambiente virtual de Python isolado e instalar requisitos:**
```bash
# Opcional (apenas se a VM não vier nativa com as libs básicas de virtualenv):
# apt-get install python3-venv python3-pip -y

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Passo 4: Sessão TMUX (Proteção Anti-Queda da Internet)
O treinamento dura horas. Se a sua internet local cair, a VM cancela a execução. Precisamos isolar o terminal num servidor de fundo através do `tmux`.

```bash
tmux new -s quantgod
```
*(Seu terminal piscará e uma barra verde surgirá no rodapé. Você está protegido agora).*

---

## Passo 5: Inicializando a Máquina e Entendendo a Resolução de Módulos (PYTHONPATH)
O `tmux` abre um terminal limpo. Repita a ativação do ambiente virtual:
```bash
cd /workspace/QuantGod_Cloud_TCNLSTM
source venv/bin/activate
```

Como o RunPod tem hardware potente, crie a árvore de diretórios vazia rapidamente para ancorar os outputs:
```bash
mkdir -p data/L2/raw data/L2/pre_processed data/L2/splits data/models data/artifacts logs/etl logs/labelling logs/optimization logs/transfer
```

### 🚨 COMO CONTORNAR ModuleNotFoundError: No module named 'src'
Ao chamar os scripts internos a partir da raiz num Linux cru, o Python pode não encontrar a pasta `src/`. 

**Solução (Faça isso antes de rodar os scripts):**
Avise ao Python para enxergar a pasta atual do RunPod como parte de suas bibliotecas injetando-a no `PYTHONPATH`:
```bash
export PYTHONPATH="${PYTHONPATH}:/workspace/QuantGod_Cloud_TCNLSTM"
```
*(Você precisará rodar essa linha novamente toda vez que reiniciar a VM ou criar um novo terminal).*

---

## Passo 6: Executando o Pipeline Completo (As 3 Etapas)

O processo ponta a ponta é composto por 3 scripts Python que rodam sequencialmente, cada um consumindo configurações de um `.yaml` diferente.

### ▶️ ETAPA 1: ETL (Extração e Transformação)
Este script baixa os Zips crus do Google Drive, cria as 16 variáveis de microestrutura (OFI, Slope, Momentum) usando processamento paralelo pesado e salva os arquivos `.parquet` escalonados via Snappy sem perda de dados.

**O Comando:**
```bash
python src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py
```

**Parâmetros de Controle (onde alterar se precisar):**
Arquivo: `src/cloud/base_model/pre_processamento/configs/cloud_config.yaml`
* `paths.rclone_mount`: De onde baixar os Zips (Padrão configurado `PROJETOS/BTC_USDT_L2_2023_2026`)
* `etl.orderbook_levels`: Nível de profundidade (200)
* `etl.resampling_interval`: Intervalo de tempo (1min)
* `etl.max_workers`: Quantos núcleos usar (Sua VM usa 14).


### ▶️ ETAPA 2: Labelling (Alvos de Compra/Venda)
Este script lê os `.parquets` recém-processados, projeta os lucros futuros e espalha os Rótulos/Classes (0=Sell, 1=Neutral, 2=Buy).

**O Comando:**
```bash
python src/cloud/base_model/labelling/run_labelling.py
```

**Parâmetros de Controle:**
Arquivo: `src/cloud/base_model/configs/labelling_config.yaml`
* `params.lookahead`: Janela no futuro para buscar lucro.
* `params.threshold_short`: Gatilho Sell (Ex: `-0.004`).
* `params.threshold_long`: Gatilho Buy (Ex: `0.004`).
*(Ele cria uma pasta única, ex: `data/L2/labelled_SELL_0004_BUY_0004_1h`)*


### ▶️ ETAPA 3: Treino (Optuna + Foundation + Specialization)
O cérebro: ele devora os dados etiquetados, hiper-otimiza camadas usando Optuna, salva os Modelos Campeões Dir / Macro e transfere o Pacote Foundation ao Google Drive de volta automaticamente.

**O Comando:**
```bash
python src/cloud/base_model/otimizacao/run_optuna.py
```

**Parâmetros de Controle:**
Arquivo: `src/cloud/base_model/otimizacao/optimization_config.yaml`
* **Importante:** Você *DEVE* usar o Nano antes de rodar, para apontar o `train_dir` e `val_dir` exatamente para o nome da `labelled_...` gerada na Etapa 2.
* `epochs`: O loop do limite de rede neural.
* `optimization.n_trials`: Quantas arquiteturas distintas o Optuna deve chutar.
* `optimization.run_specialized_after`: Se deve disparar a Subnet Especialista (True/False).

---

## 🔒 Gerenciamento da Sessão Tmux (Background)

**Como sair sem matar o modelo:**
Com o log rodando intensamente na sua tela (Seja no Optuna, ETL ou Label)...
1. Pressione `Ctrl+B` (Solte os botões).
2. Aperte rapidamente `D` (Apenas a letra D de *detach*).
O processamento passará a rodar isolado num Daemon Linux em background. Você pode fechar a janela do SSH sem medo.

**Para voltar amanhã e ver como está andando:**
```bash
tmux attach -t quantgod
```

Boa caçada, seu cluster TCN-LSTM com 16 recursos microestruturais e Optuna robusto está 100% blindado para rodar em produção! ☁️🔥
