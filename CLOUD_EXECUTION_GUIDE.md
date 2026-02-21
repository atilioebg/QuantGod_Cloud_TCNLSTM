# Guia de Execução na Cloud (RunPod) ☁️

Este guia contém o passo a passo exato para configurar o ambiente e executar o pipeline completo do **QuantGod TCNLSTM** em uma máquina Cloud (ex: RunPod), utilizando o Tmux para processos longos e o Rclone para sincronização de dados.

## 🛠️ 1. Configuração Inicial, Clone e Ambiente

Ao iniciar a máquina (Web Terminal ou SSH), execute:

```bash
cd /workspace

# 1. Clone o repositório (será solicitado seu usuário e personal access token)
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git .
# Se as credenciais forem exigidas:
# User: atilioebg
# Password: <SEU_TOKEN_AQUI> (Ex: ghp_...)

# 2. Crie e ative a Virtual Environment (venv) isolada
python -m venv venv
source venv/bin/activate

# 3. Exporte a variável de caminho do Python para referenciar a raiz
export PYTHONPATH=$PYTHONPATH:/workspace

# 4. Atualize o pip e instale as dependências
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 2. Configurar Rclone (Google Drive) e Baixar/Montar Dados

Assumindo que você tem o arquivo `rclone.conf` na raiz do projeto (como enviado localmente):

```bash
# 1. Criar pasta oficial do rclone no sistema
mkdir -p /root/.config/rclone/

# 2. Copiar o arquivo da raiz do repositório para a pasta do sistema
cp /workspace/rclone.conf /root/.config/rclone/rclone.conf

# 3. Testar a conexão (deve listar as pastas do seu Drive)
rclone lsd drive:
```

### 📥 Opção A: Download Direto via Tmux (Ideal se já processados)
Se os dados já estão rotulados ou processados e você quer apenas baixar:
```bash
tmux new -s download_dataset
rclone copy drive:PROJETOS/L2/pre_processed /workspace/data/L2/pre_processed -P
# Para sair deixando rodar: Pressione Ctrl + B e depois D
```

---

## 🔄 3. Processamento Completo de ETL e Labelling na Cloud

Se precisar rodar o processo pesado na Cloud:

### Executando o Pré-processamento (ETL)
*Isso deve ser feito caso os dados no Drive sejam .zip brutos e você processe na Cloud.*

```bash
# Certifique-se de estar na raiz do projeto
cd /workspace
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace

# Iniciar uma sessão tmux
tmux new -s pipeline_god

# Executar a orquestração do pipeline ETL (Ajuste o arquivo de config yaml se precisar)
python src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py src/cloud/base_model/pre_processamento/configs/cloud_config.yaml

# Sair do tmux e deixar rolando: Ctrl + B, soltar, e apertar D
```

### Como recuperar o terminal Tmux:
```bash
# Listar as sessões ativas
tmux ls

# Reconectar à sessão do pipeline
tmux attach -t pipeline_god
```

### Testando o Pré-processamento
Após concluir o ETL, teste a integridade dos dados gerados:
```bash
pytest tests/test_cloud_etl_output.py
pytest tests/test_preprocessed_quality.py
```

---

## 🏷️ 4. Labelling (Rotulagem)

Após o ETL passar nos testes, aplicamos as rotulagens de Buy/Sell:

```bash
cd /workspace
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace

# Execute a rotulagem passando a configuração desejada (ex: labelling_config.yaml)
python src/cloud/base_model/labelling/run_labelling.py src/cloud/base_model/labelling/labelling_config.yaml

# Testando o resultado da Rotulagem:
# Defina a variável para o diretório que acabou de ser gerado (ex: labelled_SELL_0001_BUY_0001_2h)
export LABELLED_DIR="data/L2/labelled_SELL_0001_BUY_0001_2h"
pytest tests/test_labelling_output.py
```

---

## 🪚 5. Split do Dataset Train/Val/Test

Após aprovação nos testes, vamos separar os dados (70/20/10):

```bash
cd /workspace
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace

# Separa as imagens indicando o diretório rotulado
python src/cloud/base_model/treino/split_dataset.py data/L2/labelled_SELL_0001_BUY_0001_2h
```

---

## 🧠 6. Treinamento na GPU e Optuna (Uploads)

Com os dados de Treino/Val/Test criados, é iniciada a otimização e o treino final:

```bash
cd /workspace
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace

# Opcional: Busca por Hiperparâmetros (pode ser executado no Tmux também)
python src/cloud/base_model/otimizacao/run_optuna.py src/cloud/base_model/otimizacao/optimization_config.yaml

# Treinamento do Modelo Híbrido Final (TCN+LSTM)
# Se os melhores hiperparâmetros foram encontrados, o sistema carregará o best_params.json
python src/cloud/base_model/treino/run_training.py src/cloud/base_model/treino/training_config.yaml
```

---

## 💾 7. Backup Automático: Logs e L2 para o Google Drive

Para assegurar todo log e dado modificado sejam gravados permanentemente no seu drive:

```bash
# 1. Copiar Logs de Processamento e ETL
rclone --config /workspace/rclone.conf copy /workspace/logs/labelling drive:PROJETOS/L2/logs/labelling -P
rclone --config /workspace/rclone.conf copy /workspace/logs/etl drive:PROJETOS/L2/logs/etl -P

# 2. Copiar todo o Dataset Processado via Tmux
tmux new -s upload_drive
rclone --config /workspace/rclone.conf copy /workspace/data/L2 drive:PROJETOS/L2 -P
# Sair (Ctrl+B depois D) para liberar a janela
```

> **📌 Validação Pós-Cloud no Ambiente Local:**
> No seu terminal Windows local (PowerShell), monte a pasta se desejar realizar uma nova bateria de validações local, assegurando que os relatórios do rclone bateram perfeitamente na nuvem!
> ```powershell
> .\rclone mount drive: Z: --vfs-cache-mode full --config "c:\Users\Atilio\Desktop\PROJETOS\PESSOAL\QuantGod\rclone.conf"
> ```
