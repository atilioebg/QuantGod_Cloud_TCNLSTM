# Guia de Execu├º├úo na Nuvem (RunPod) - Pipeline TCN-LSTM Ponta a Ponta

Este guia fornece o passo a passo definitivo para preparar um ambiente RunPod do absoluto zero e rodar individualmente os 3 scripts que comp├Áem o pipeline final (ETL de 16 vari├íveis -> Labelling -> Treinamento Optuna / Funda├º├úo).

Cole os comandos linha a linha no Web Terminal ou via SSH do seu RunPod rec├®m-criado.

---

## Passo 1: Prepara├º├úo do Sistema Operacional e Ferramentas B├ísicas (Ubuntu)
Geralmente os templates do RunPod v├¬m em modo root e podem estar sem alguns pacotes vitais para edi├º├úo de texto e ger├¬ncia de sess├Áes.

```bash
apt-get update && apt-get install -y nano tmux pciutils wget curl unzip zip htop sudo software-properties-common rsync
```

## Passo 2: Instala├º├úo do Rclone e Montagem do Google Drive

**2.1 - Baixar e instalar a vers├úo oficial do Rclone para Linux via script:**
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

**2.2 - Criar o diret├│rio raiz para o Workspace:**
Aqui viver├úo os dados persistentes no volume de Network.
```bash
cd /workspace
mkdir -p data
```

**2.3 - Configurar o token do Google Drive:**
Crie o arquivo de configura├º├úo do Rclone usando o `nano`:
```bash
nano /workspace/rclone.conf
```
Dentro do Nano, cole a configura├º├úo do seu token:
```ini
[drive]
type = drive
scope = drive
token = {"access_token":"ya29..."} # Substitua pela sua linha completa do token
```
*Salve e feche (`Ctrl+O`, `Enter`, `Ctrl+X`).*

## Passo 3: Clonagem do Reposit├│rio e Ambiente Virtual Python
Garanta que sua pasta raiz ser├í `/workspace/`. O c├│digo fonte, o virtual env e os logs devem viver nela.

**3.1 - Clonar reposit├│rio e selecionar a branch correta:**
```bash
cd /workspace
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git
cd QuantGod_Cloud_TCNLSTM
git checkout tcn_lstn_features
```

**3.2 - Criar ambiente virtual de Python isolado e instalar requisitos:**
```bash
# Opcional (apenas se a VM n├úo vier nativa com as libs b├ísicas de virtualenv):
# apt-get install python3-venv python3-pip -y

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest-xdist  # Necess├írio para os 12 workers paralelos
```

## Passo 4: Sess├úo TMUX (Prote├º├úo Anti-Queda da Internet)
O treinamento dura horas. Se a sua internet local cair, a VM cancela a execu├º├úo. Precisamos isolar o terminal num servidor de fundo atrav├®s do `tmux`.

```bash
tmux new -s quantgod
```
*(Seu terminal piscar├í e uma barra verde surgir├í no rodap├®. Voc├¬ est├í protegido agora).*

---

## Passo 5: Inicializando a M├íquina e Entendendo a Resolu├º├úo de M├│dulos (PYTHONPATH)
O `tmux` abre um terminal limpo. Repita a ativa├º├úo do ambiente virtual:
```bash
cd /workspace/QuantGod_Cloud_TCNLSTM
source venv/bin/activate
```

Como o RunPod tem hardware potente, crie a ├írvore de diret├│rios vazia rapidamente para ancorar os outputs:
```bash
mkdir -p data/L2/raw data/L2/pre_processed data/L2/splits data/models data/artifacts logs/etl logs/labelling logs/optimization logs/transfer logs/tests
```

### ­ƒÜ¿ COMO CONTORNAR ModuleNotFoundError: No module named 'src'
Ao chamar os scripts internos a partir da raiz num Linux cru, o Python pode n├úo encontrar a pasta `src/`. 

**Solu├º├úo (Fa├ºa isso antes de rodar os scripts):**
Avise ao Python para enxergar a pasta atual do RunPod como parte de suas bibliotecas injetando-a no `PYTHONPATH`:
```bash
export PYTHONPATH="${PYTHONPATH}:/workspace/QuantGod_Cloud_TCNLSTM"
```
*(Voc├¬ precisar├í rodar essa linha novamente toda vez que reiniciar a VM ou criar um novo terminal).*

---

## Passo 6: Executando o Pipeline Completo (As 3 Etapas)

O processo ponta a ponta ├® composto por 3 scripts Python que rodam sequencialmente, cada um consumindo configura├º├Áes de um `.yaml` diferente.

### ÔûÂ´©Å ETAPA 1: ETL (Extra├º├úo e Transforma├º├úo)
Este script baixa os Zips crus do Google Drive, cria as 16 vari├íveis de microestrutura (OFI, Slope, Momentum) usando processamento paralelo pesado e salva os arquivos `.parquet` escalonados via Snappy sem perda de dados.

**O Comando:**
```bash
python src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py
```

**Par├ómetros de Controle (onde alterar se precisar):**
Arquivo: `src/cloud/base_model/pre_processamento/configs/cloud_config.yaml`
* `paths.rclone_mount`: De onde baixar os Zips (Padr├úo configurado `PROJETOS/BTC_USDT_L2_2023_2026`)
* `etl.orderbook_levels`: N├¡vel de profundidade (200)
* `etl.resampling_interval`: Intervalo de tempo (1min)
* `etl.max_workers`: Quantos n├║cleos usar (Sua VM usa 14).

**Verifica├º├úo e Backup (Opcional):**
```bash
# Validar se os Parquets foram gerados corretamente
pytest tests/etl/test_cloud_etl_output.py -v -n 12

# O log detalhado ser├í salvo automaticamente em:
# /workspace/QuantGod_Cloud_TCNLSTM/logs/tests/last_run.log
```

# Backup dos dados pré-processados para o Google Drive
rclone copy data/L2/pre_processed drive:PROJETOS/PRE_PROCESSED_L2_2023_2026_1_MINUTE_18_FEATURES/ --config rclone.conf -P
```


### ÔûÂ´©Å ETAPA 2: Labelling (Alvos de Compra/Venda)
Este script l├¬ os `.parquets` rec├®m-processados, projeta os lucros futuros e espalha os R├│tulos/Classes (0=Sell, 1=Neutral, 2=Buy).

**O Comando:**
```bash
python src/cloud/base_model/labelling/run_labelling.py
```

**Par├ómetros de Controle:**
Arquivo: `src/cloud/base_model/configs/labelling_config.yaml`
* `params.lookahead`: Janela no futuro para buscar lucro.
* `params.threshold_short`: Gatilho Sell (Ex: `-0.004`).
* `params.threshold_long`: Gatilho Buy (Ex: `0.004`).
*(Ele cria uma pasta ├║nica, ex: `data/L2/labelled_SELL_0004_BUY_0004_1h`)*

**Verifica├º├úo e Backup (Opcional):**
```bash
# Validar distribui├º├úo de classes e integridade dos r├│tulos
pytest tests/labelling/test_labelling_output.py -v -n 12

# Backup dos dados rotulados para o Google Drive
rclone copy data/L2/labelled_* drive:PROJETOS/LABELLED_L2_2023_2026_1_MINUTE_18_FEATURES/ --config rclone.conf -P
```


### ÔûÂ´©Å ETAPA 3: Treino (Optuna + Foundation + Specialization)
O c├®rebro: ele devora os dados etiquetados, hiper-otimiza camadas usando Optuna, salva os Modelos Campe├Áes Dir / Macro e transfere o Pacote Foundation ao Google Drive de volta automaticamente.

**O Comando:**
```bash
python src/cloud/base_model/otimizacao/run_optuna.py
```

**Par├ómetros de Controle:**
Arquivo: `src/cloud/base_model/otimizacao/optimization_config.yaml`
* **Importante:** Voc├¬ *DEVE* usar o Nano antes de rodar, para apontar o `train_dir` e `val_dir` exatamente para o nome da `labelled_...` gerada na Etapa 2.
* `epochs`: O loop do limite de rede neural.
* `optimization.n_trials`: Quantas arquiteturas distintas o Optuna deve chutar.
* `optimization.run_specialized_after`: Se deve disparar a Subnet Especialista (True/False).

---

## ­ƒöÆ Gerenciamento da Sess├úo Tmux (Background)

**Como sair sem matar o modelo:**
Com o log rodando intensamente na sua tela (Seja no Optuna, ETL ou Label)...
1. Pressione `Ctrl+B` (Solte os bot├Áes).
2. Aperte rapidamente `D` (Apenas a letra D de *detach*).
O processamento passar├í a rodar isolado num Daemon Linux em background. Voc├¬ pode fechar a janela do SSH sem medo.

**Para voltar amanh├ú e ver como est├í andando:**
```bash
tmux attach -t quantgod
```

Boa ca├ºada, seu cluster TCN-LSTM com 16 recursos microestruturais e Optuna robusto est├í 100% blindado para rodar em produ├º├úo! Ôÿü´©Å­ƒöÑ

