# Guia de Execução na Nuvem (RunPod) - Pipeline TCN-LSTM Ponta a Ponta

Este guia fornece o passo a passo definitivo para preparar um ambiente RunPod do absoluto zero e rodar individualmente os scripts do pipeline (ETL -> Labelling -> Treinamento Optuna / Fundação / Especialista).

Cole os comandos linha a linha no Web Terminal ou via SSH do seu RunPod recém-criado.

---

## Passo 1: Preparação do Sistema Operacional (Ubuntu)

```bash
apt-get update && apt-get install -y nano tmux pciutils wget curl unzip zip htop sudo software-properties-common rsync
```

## Passo 2: Instalação do Rclone e Clonagem do Repositório

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

## Passo 3: O Botão do Pânico (Limpeza Mestra da VM)

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

## Passo 4: Iniciando Pipeline (Via Tmux)

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

## Passo 5: Executar o Fluxo (Comandos Diretos)

### ▶️ ETAPA 1: Pré-Processamento (ETL de 16 Atributos)
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
rclone copy data/L2/pre_processed drive:PROJETOS/PRE_PROCESSED_L2_2023_2026_1_MINUTE_18_FEATURES/ --config rclone.conf -P
```

### ▶️ ETAPA 2: Labelling (Rótulos do Futuro)
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
rclone copy data/L2/labelled_* drive:PROJETOS/LABELLED_L2_2023_2026_1_MINUTE_18_FEATURES/ --config rclone.conf -P
```

### ▶️ ETAPA 3: Treinamento Pesado (Finetuning/Fundação)
**Objetivo:** Rodar a busca Optuna, encontrar o Top 1, salvar o modelo campeão e gerar validações matemáticas.
**ATENÇÃO:** Abra o `src/cloud/base_model/otimizacao/optimization_config.yaml` e atualize os caminhos do `train_dir` e `val_dir` para apontarem para a exata pasta gerada na "Etapa 2" (ex: `data/L2/splits_labelled_.../train`).
```bash
python src/cloud/base_model/otimizacao/run_optuna.py
```
