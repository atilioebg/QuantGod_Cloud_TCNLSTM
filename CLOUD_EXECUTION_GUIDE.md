# Guia de Execução na Nuvem (RunPod) - Instalação do Zero

Este guia fornece as instruções otimizadas para preparar um ambiente RunPod virgem (nova VM e novo Storage Network) de ponta a ponta, desde a instalação de pacotes essenciais até a execução do pipeline de treinamento TCN-LSTM com 16 recursos de microestrutura.

Estes comandos devem ser colados no Web Terminal ou via SSH do RunPod recém-criado.

## Passo 1: Preparação do Sistema Operacional (Ubuntu) e Ferramentas Básicas
Os templates do RunPod iniciam em modo root e podem carecer de pacotes vitais para edição de texto e gerência de janelas.

```bash
apt-get update && apt-get install -y nano tmux pciutils wget curl unzip zip htop sudo software-properties-common rsync
```

## Passo 2: Instalação e Configuração do Rclone
O Rclone é necessário para montar e acessar o Google Drive onde residem os dados L2.

**2.1 - Baixar e instalar o Rclone:**
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

**2.2 - Criar o diretório raiz do Workspace:**
Aqui viverão os dados persistentes no volume de Network.
```bash
cd /workspace
mkdir -p data logs
```

**2.3 - Configurar o token do Google Drive:**
Crie o arquivo de configuração do Rclone usando o `nano`:
```bash
nano /workspace/rclone.conf
```
Dentro do editor Nano, cole o bloco abaixo substituindo a string inteira do `token` pelo seu token válido:
```ini
[drive]
type = drive
scope = drive
token = {"access_token":"ya29..."} # Cole a linha toda do seu token
```
*Salve e feche o Nano (`Ctrl+O`, `Enter`, `Ctrl+X`).*

## Passo 3: Clonando o Repositório e Configurando o Ambiente Python
Garanta que a pasta raiz será `/workspace`. O código fonte, o virtual environment e os logs viverão dentro dela.

**3.1 - Clonar e acessar a branch:**
```bash
cd /workspace
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git
cd QuantGod_Cloud_TCNLSTM
git checkout tcn_lstn_features
```

**3.2 - Criar o Virtual Environment e instalar as dependências:**
```bash
# Opcional (apenas se a VM não vier nativa com as libs básicas de virtualenv):
# apt-get install python3-venv python3-pip -y

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Passo 4: Iniciando a Proteção Anti-Queda (TMUX)
O treinamento do Optuna pode durar horas. Se a conexão SSH oscilar, o processo morre. O `tmux` encapsula o terminal protegendo a execução em segundo plano.

```bash
tmux new -s quantgod
```
*(Seu terminal piscará e uma barra verde aparecerá no rodapé. Você está na sessão isolada).*

## Passo 5: Inicializando a Máquina e Rodando o Pipeline

O `tmux` abre um prompt limpo. Reative o ambiente virtual:
```bash
cd /workspace/QuantGod_Cloud_TCNLSTM
source venv/bin/activate
```

Como o RunPod possui hardware potente, crie todas as pastas padrão necessárias para a estrutura do projeto não falhar por arquivos inexistentes:
```bash
mkdir -p data/L2/raw data/L2/pre_processed data/L2/splits data/models data/artifacts logs/etl logs/labelling logs/optimization logs/transfer
```

### 🚨 Como Contornar Erros de Importação (ModuleNotFoundError)
Como o pacote `src` não está instalado globalmente (`pip install -e .`) e o código é chamado a partir da raiz da pasta, o terminal Linux virgem pode cuspir um `ModuleNotFoundError: No module named 'src'`. 

Você deve setar a variável de ambiente **PYTHONPATH** dizendo para a máquina onde a biblioteca do seu código raiz vive:
```bash
export PYTHONPATH="${PYTHONPATH}:/workspace/QuantGod_Cloud_TCNLSTM"
```

### ▶️ Iniciando a Execução (ETL)
O comando que puxa os dados do Driver, pré-processa as 16 Features, aplica as transformações e salva os parquets é:
```bash
python src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py
```

*(Ou usando a "solução à prova de balas" chamando como um módulo caso o PYTHONPATH não fixe o erro das pastas:*
```bash
python -m src.cloud.base_model.pre_processamento.orchestration.run_pipeline
```
*)*

## Gerenciando a Sessão Tmux (Básicos)

**Como Sair da Máquina sem Matar o Processo (Detach):**
Quando o comando estiver rodando na tela verde soltando os logs:
1. Pressione `Ctrl+B`.
2. Solte todas as teclas.
3. Aperte rapidamente `D` (Apenas a letra D de 'detach').
Você voltará ao shell original da máquina e pode fechar o SSH em paz.

**Para voltar amanhã e ver o progresso (Attach):**
```bash
tmux attach -t quantgod
```
