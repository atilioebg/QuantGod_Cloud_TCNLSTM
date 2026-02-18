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
- **Configuração**: `src/cloud/pre_processamento/configs/cloud_config.yaml`
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
- **Configuração**: `src/cloud/labelling/labelling_config.yaml`
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
- **Configuração**: `src/cloud/otimizacao/optimization_config.yaml`
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
- **Configuração**: `src/cloud/treino/training_config.yaml`
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

### 1. A Pegadinha do Caminho (Z:/ vs /workspace/) 📂
O arquivo `cloud_config.yaml` precisa ser ajustado conforme o ambiente:
- **Local (Windows)**: `rclone_mount: "Z:/PROJETOS/..."`
- **Cloud (Linux/RunPod)**: `rclone_mount: "/workspace/gdrive/..."`

### 2. Consistência ob500 vs ob200
O pipeline aplica um **Hard Cut** automático para 200 níveis. Isso garante que, independentemente da profundidade do arquivo original (2023 vs 2026), o output terá **exatamente as mesmas colunas**, evitando erros no treinamento.

### 3. Erro: `path not found` ou `Z:\...` inexistente
- Verifique se o Rclone está rodando (Passo 2).
- Se estiver no Linux, verifique se o caminho no `cloud_config.yaml` aponta para `/workspace/gdrive/...`.

### 4. Erro: `Out of Memory (OOM)`
- Reduza o `batch_size` nos arquivos de configuração `.yaml`.
- No ETL, reduza o número de workers em `run_pipeline.py`.
