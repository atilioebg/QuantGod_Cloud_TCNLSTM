# QuantGod Cloud Infrastructure ☁️

Este diretório contém o pipeline modular de Processamento e Treinamento do QuantGod, projetado para escalar horizontalmente em instâncias de nuvem (RunPod, GCP, AWS) ou rodar localmente para desenvolvimento.

---

## 🛠️ Configuração do Ambiente

Antes de iniciar, certifique-se de satisfazer as dependências.

### 1. Instalação de Bibliotecas
Na raiz do projeto:
```powershell
pip install -r requirements.txt
```

### 2. Conexão com Dados (Rclone) 🔌
O pipeline **não baixa** os terabytes de dados para o disco local. Ele usa **streaming** via mount de disco. Você precisa "montar" o Google Drive do projeto.

#### **Opção A: Windows (Local / Dev)**
Use o executável `rclone.exe` já incluído na raiz do projeto.
1. Abra um terminal **PowerShell como Administrador**.
2. Execute o comando para montar o drive na letra `Z:`:
   ```powershell
   .\rclone.exe mount drive: Z: --vfs-cache-mode full --config rclone.conf
   ```
   *⚠️ Mantenha esta janela do terminal aberta enquanto estiver trabalhando.*

#### **Opção B: Linux (Cloud / RunPod)**
Em instâncias Linux, montamos em `/workspace/gdrive`.
```bash
# Insta-le o rclone se necessário
curl https://rclone.org/install.sh | sudo bash

# Configure (se ainda não tiver o rclone.conf)
rclone config

# Crie a pasta e monte em background
mkdir -p /workspace/gdrive
rclone mount drive: /workspace/gdrive --vfs-cache-mode full --allow-other &
```

---

## 🚀 Pipeline de Execução Passo a Passo

Siga esta ordem rigorosa para reproduzir o ciclo de vida do modelo.

### 1. Pré-processamento (ETL) 🧹
Transforma os arquivos brutos ZIP (Bybit L2) em arquivos Parquet otimizados e limpos. 
- **Multi-Ano**: O pipeline realiza busca **recursiva** em subpastas (2023, 2024, etc.).
- **Compatibilidade**: Suporta arquivos `ob500` e `ob200` aplicando um *Hard Cut* automático para 200 níveis.
- **Configuração**: `src/cloud/base_model/configs/master_config.yaml`
- **Output**: `data/L2/pre_processed/*.parquet`
- **Comando**:
  ```powershell
  python -m src.cloud.pre_processamento.orchestration.run_pipeline
  ```
- **Validação e Qualidade**:
  ```powershell
  pytest tests/test_cloud_etl_output.py
  pytest tests/test_preprocessed_quality.py
  ```

### 2. Rotulagem (Labelling) 🏷️
Aplica a lógica econômica (Thresholds Assimétricos) para criar os alvos (`target`): 0 (Sell), 1 (Neutral), 2 (Buy).
- **Configuração**: `src/cloud/base_model/configs/master_config.yaml`
- **Output**: `data/L2/labelled/*.parquet`
- **Comando**:
  ```powershell
  python src/cloud/labelling/run_labelling.py
  ```
- **Validação**: Verifica se as classes não estão zeradas:
  ```powershell
  pytest tests/test_labelling_output.py
  ```

### 3. Otimização de Hiperparâmetros (Optuna) 🎯
Utiliza busca Bayesiana para encontrar a melhor arquitetura do Transformer (n_heads, layers, dropout, lr), maximizando o **F1-Score Ponderado**.
- **Configuração**: `src/cloud/base_model/configs/master_config.yaml`
- **Comando**:
  ```powershell
  python src/cloud/otimizacao/run_optuna.py
  ```
- **Output**: 
  - `src/cloud/otimizacao/best_params.json` (Melhores configs).
  - `optuna_study.db` (Histórico da otimização).

#### 📊 Dashboard em Tempo Real
Para visualizar gráficos de convergência e importância de parâmetros:
```powershell
optuna-dashboard sqlite:///optuna_study.db
# Acesse no navegador: http://127.0.0.1:8080/
```

### 4. Treinamento Final (Fine-Tuning) 🧠
Treina o modelo `QuantGodModel` definitivo usando os melhores parâmetros encontrados pelo Optuna.
- **Configuração**: `src/cloud/base_model/configs/master_config.yaml`
- **Input**: Lê automaticamente `best_params.json` se disponível (ou usa o config padrão).
- **Comando**:
  ```powershell
  python src/cloud/treino/run_training.py
  ```
- **Output**: `data/models/quantgod_cloud_model.pth`

---

## 📂 Logs e Auditoria
O sistema mantem logs detalhados para debugging e auditoria de performance.

| Pasta | Conteúdo | Importância |
| :--- | :--- | :--- |
| `logs/etl/` | Arquivos processados, erros de leitura, uso de CPU. | Alta (Integridade) |
| `logs/labelling/` | Contagem de classes (Buy/Sell), arquivos vazios. | Alta (Balanceamento) |
| `logs/optimization/` | Loss, F1 e Acurácia de cada trial do Optuna. | Média (Performance) |
| `logs/training/` | Evolução da Loss e F1 por época do treino final. | Alta (Convergência) |

---

## ⚡ Performance e Hardware Recomendado

O processamento L2 é intensivo em CPU devido à reconstrução do Orderbook segundo a segundo (1000ms).

*   **Processamento de ob500 (2023)**: Exige significativamente mais CPU que o ob200.
*   **Instância Recomendada (RunPod/Cloud)**: 
    *   Mínimo: **4 vCPUs** / **16GB RAM**.
    *   Ideal: **8+ vCPUs** para paralelismo máximo no ETL.
*   **GPU**: Necessária apenas para as etapas 3 (Optimization) e 4 (Training). Uma RTX 3090/4090 ou instâncias de A100 são recomendadas para velocidade.

---

## 🆘 Troubleshooting & Checklist Final

### 1. A Pegadinha do Caminho (Drive Remoto) 📂
O arquivo `master_config.yaml` usa o prefixo `drive:` para o Rclone. Certifique-se de que o `rclone.conf` está correto.
- Se estiver no Linux, verifique se o mount está apontando para o local correto se estiver usando mount, mas o pipeline agora prefere streaming direto via `rclone copy/ls`.

### 2. Consistência ob500 vs ob200
O pipeline aplica um **Hard Cut** automático para 200 níveis. Isso garante que, independentemente da profundidade do arquivo original (2023 vs 2026), o output terá **exatamente as mesmas colunas**, evitando erros no treinamento.

### 3. Erro: `path not found` ou `Z:\...` inexistente
- Verifique se o Rclone está rodando (Passo 2).
- Se estiver no Linux, verifique se o caminho no `cloud_config.yaml` aponta para `/workspace/gdrive/...`.

### 4. Erro: `Out of Memory (OOM)`
- Reduza o `batch_size` nos arquivos de configuração `.yaml`.
- No ETL, reduza o número de workers em `run_pipeline.py`.

---

## 🚀 Passo a Passo para rodar na Cloud (RUNPOD)

Este guia prático fornece todos os comandos necessários para executar o pipeline completo em uma instância RunPod, garantindo persistência e organização.

### 1. Preparação Inicial do Terminal
Após entrar no **Web Terminal** do RunPod, prepare o ambiente básico:

```bash
cd /workspace

# Sincronizar o código mais recente
git pull origin main
# Credenciais (se solicitado):
# User: atilioebg
# Token: <SEU_GITHUB_TOKEN_AQUI>

# Ativar ambiente e caminhos
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace
```

### 2. Execução do Pré-processamento (ETL)
Usaremos o `tmux` para garantir que o processamento continue mesmo se você fechar o navegador.

```bash
# Criar uma sessão persistente
tmux new -s pipeline_god

# Dentro do tmux, prepare o ambiente novamente (necessário por sessão)
cd /workspace
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace

# Disparar o motor de ETL
python3 src/cloud/pre_processamento/orchestration/run_pipeline.py
```

*   **Comandos úteis do tmux:**
    *   `Ctrl + B` depois `D`: Sair do terminal sem interromper o processo (Detach).
    *   `tmux ls`: Listar sessões ativas.
    *   `tmux attach -t pipeline_god`: Retornar ao terminal do pipeline.

### 3. Validação do ETL
Sempre valide os dados antes de prosseguir:

```bash
pytest tests/test_cloud_etl_output.py
```

### 4. Execução da Rotulagem (Labelling)
Com os dados limpos, aplicamos as regras de estratégia para criar os alvos de treino:

```bash
# Garante que as configurações de thresholds estão atualizadas
git pull origin main

source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace
python3 src/cloud/labelling/run_labelling.py

# Validar integridade dos labels
pytest tests/test_labelling_output.py
```

### 5. Backup e Sincronização com Google Drive
Não confie no armazenamento efêmero do Pod. Faça o backup dos logs e dados processados:

```bash
# Copiar logs de auditoria
rclone --config /workspace/rclone.conf copy /workspace/logs/labelling/labelling_processing.log drive:PROJETOS/
rclone --config /workspace/rclone.conf copy /workspace/logs/etl/etl_processing.log drive:PROJETOS/

# Backup total da pasta L2 (Data) para o Drive (use tmux para processos longos)
tmux new -s upload_drive
rclone --config /workspace/rclone.conf copy /workspace/data/L2 drive:PROJETOS/L2 -P
```

### 6. Validação Local (Opcional - Pós-Cloud)
Após copiar os dados para o Drive, você pode conferir a integridade na sua máquina local montando o Drive como unidade `Z:`:

```powershell
# Comando para rodar no terminal local (Windows)
.\rclone mount drive: Z: --vfs-cache-mode full --config "c:\Users\Atilio\Desktop\PROJETOS\PESSOAL\QuantGod\rclone.conf"
```
*Com a unidade montada, você pode rodar os mesmos arquivos de `pytest` localmente apontando para `Z:\`.*

---

## ⚡ TREINO GPU CLOUD - STORAGE NOVO

Este guia é destinado a instâncias de GPU zeradas (como RunPod RTX 4090/A100 com storage vazio), onde o foco é baixar o dataset rotulado para o disco local (`/workspace/data`) para máxima performance de I/O durante o treino.

### 1. Ambiente e Dependências
Execute estes comandos para clonar o projeto e preparar o ambiente virtual:

```bash
cd /workspace

# Clonar o repositório diretamente na raiz do workspace
git clone https://github.com/atilioebg/QuantGod_Cloud.git .

# Setup do Ambiente Virtual
python -m venv venv
source venv/bin/activate

# Instalação Otimizada (Totalmente corrigida)
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configurações de Nuvem (Rclone)
Configure a ponte com o Google Drive usando o arquivo integrado ao repositório:

```bash
# Preparar o diretório de configuração do sistema
mkdir -p /root/.config/rclone/

# Migrar a configuração do repositório para o sistema
cp /workspace/rclone.conf /root/.config/rclone/rclone.conf

# Validar Conexão
rclone lsd drive:
```

### 3. Recuperação de Dataset (Alta Performance)
Para treinos pesados, não use o `mount`. Baixe o dataset rotulado diretamente para o disco NVMe da instância usando `tmux`:

```bash
# Iniciar sessão persistente
tmux new -s download_data

# Executar o download do dataset +0.4% / -0.4% em 2h (Foundation Model)
# Ajuste o caminho se necessário
mkdir -p /workspace/data/L2/labelled/
rclone copy drive:PROJETOS/L2/labelled/labelled_SELL_0004_BUY_0004_2h /workspace/data/L2/labelled/labelled_SELL_0004_BUY_0004_2h -P

# Sair do tmux (Background): Ctrl + B, depois D
```

### 4. Iniciar Treinamento Final
Após o término do download, execute o treinamento:

```bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace
python3 src/cloud/treino/run_training.py
```
