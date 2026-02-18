# QuantGod Cloud Infrastructure ☁️

Este diretório contém o pipeline de ETL (Extract, Transform, Load) projetado para processar terabytes de dados de Orderbook (L2) na nuvem (RunPod) de forma eficiente, utilizando streaming de dados e otimização de memória.

---

## 📂 Estrutura de Arquivos e Pastas

### 1. `configs/` (Configurações)
Arquivos YAML que definem o comportamento do pipeline.
*   **`cloud_config.yaml`**: Configuração oficial para produção no RunPod. Aponta para o diretório de dados montado via `rclone`.
*   **`test_local.yaml`**: Configuração para testes em ambiente de desenvolvimento. Aponta para pastas locais (`data/L2/raw/l2_samples`).

**Parâmetros Principais:**
*   `paths.rclone_mount`: Caminho do mount do Google Drive.
*   `paths.processed_output`: Destino dos arquivos `.parquet`.
*   `etl.orderbook_levels`: Nível do **Hard Cut** (Ex: 200).
*   `features.apply_zscore`: Ativa/Desativa a normalização estatística.

### 2. `etl/` (Módulos de Processamento)
O motor do processamento, dividido em responsabilidades modulares:

*   **`extract.py`**: Implementa a lógica **Zero-Copy**. Ele abre os ZIPs diretamente do mount e lê o conteúdo (JSON/CSV) linha por linha em buffer de memória, sem nunca descompactar arquivos no disco físico do RunPod.
*   **`transform.py`**: O cérebro do pipeline.
    *   Reconstrói o Orderbook a partir de snapshots e deltas.
    *   Aplica o **Hard Cut 200** (mantém estritamente os top 200 níveis).
    *   Realiza amostragem temporal (1s ticks) e resampling (1min OHLCV).
    *   Calcula Micro-Price, Spread e IOBI.
    *   Aplica **Stationarity Fix** (Log-Returns para preços e Log1p para volume).
*   **`load.py`**: Gerencia a persistência. Utiliza o formato **Apache Parquet** com compressão **Snappy** para garantir leitura ultra-rápida durante o treino do modelo.
*   **`validate.py`**: Garante a qualidade do dado. Verifica se há NaNs, valores infinitos, se a ordem cronológica está correta e se existem "gaps" de tempo excessivos.

### 3. `orchestration/` (Coordenação)
*   **`run_pipeline.py`**: O ponto de entrada. Ele coordena o fluxo entre todos os módulos acima. Suporta a passagem de arquivos de config via terminal:
    `python -m src.cloud.pre_processamento.orchestration.run_pipeline src/cloud/pre_processamento/configs/test_local.yaml`

### 4. `setup_cloud.sh` (Automação de Ambiente)
Script bash para preparar a instância Linux (RunPod).
*   Instala pacotes do sistema (`rclone`, `python3-pip`).
*   Cria o ambiente virtual `.venv`.
*   Instala as dependências de Python.
*   Cria a árvore de diretórios oficial (`data/L2/pre_processed`, `data/artifacts`, etc.).

---

## 🚀 Como Usar na Cloud (RunPod)

Para rodar o processamento completo dos anos 2023-2026, siga estes passos ajustados para o ambiente Ubuntu 24.04:

### Passo 1: Preparar a máquina
```bash
# Execute o script de setup (ele criará .venv, pastas de dados/logs e instalará dependências)
chmod +x src/cloud/pre_processamento/setup_cloud.sh
./src/cloud/pre_processamento/setup_cloud.sh
```

### Passo 2: Configurar e Ativar o Rclone
No Linux (RunPod), o mount é feito em um diretório do sistema:

1. **Configurar Credenciais**:
   * O arquivo `rclone.conf` no storage provavelmente está vazio.
   * `nano /workspace/rclone.conf`
   * Cole o conteúdo do seu `rclone.conf` local (que começa com `[drive]`).
   * Salve (Ctrl+O, Enter) e saia (Ctrl+X).

2. **Montar o Drive**:
   ```bash
   mkdir -p /workspace/mnt/gdrive
   rclone mount drive: /workspace/mnt/gdrive --config /workspace/rclone.conf --daemon --vfs-cache-mode writes
   ```
   *Verifique com `ls /workspace/mnt/gdrive` se suas pastas apareceram.*

3. **Ajustar Caminhos**:
   * No arquivo `src/cloud/pre_processamento/configs/cloud_config.yaml`, verifique se o `rclone_mount` aponta corretamente para a pasta montada. Ex: `rclone_mount: "/workspace/mnt/gdrive/PROJETOS/BTC_USDT_L2_2023_2026"`.

### Passo 3: Rodar o Processamento (Modo Persistente)
Como o processamento pode levar horas, use o `tmux` para garantir que o script continue rodando mesmo se você fechar o navegador.

1. **Entrar no tmux**:
   ```bash
   tmux new -s pilar_etl
   ```

2. **Ativar Ambiente e Disparar**:
   ```bash
   source .venv/bin/activate
   export PYTHONPATH=$PYTHONPATH:/workspace
   python3 src/cloud/pre_processamento/orchestration/run_pipeline.py
   ```

3. **Comandos Úteis do tmux**:
   * **Desconectar (Sair sem parar)**: `Ctrl + B`, solte, e aprete `D`.
   * **Reconectar**: `tmux attach -t pilar_etl`.
   * **Navegar nos logs**: `Ctrl + B`, solte, e aprete `[` para usar as setas (esc para sair).

---

## 💻 Como Usar Local (Windows)

#### 1. Montar o Google Drive
Como o WinFSP já está instalado, use o `rclone.exe` na raiz:
```powershell
.\rclone.exe mount drive: Z: --vfs-cache-mode full --config rclone.conf
```
*Mantenha o terminal aberto.*

#### 2. Rodar Testes Locais
```bash
source .venv/bin/activate
python -m src.cloud.pre_processamento.orchestration.run_pipeline src/cloud/pre_processamento/configs/test_local.yaml
```

---

## 🛠️ Requisitos Técnicos (`requirements.txt`)
O pipeline depende de:
*   `polars` / `pandas`: Processamento de dados de alta performance.
*   `pyarrow`: Engine para escrita de Parquet.
*   `scikit-learn`: Para aplicação do `StandardScaler` (Z-Score).
*   `tqdm`: Barras de progresso para monitoramento de grandes volumes.
*   `pyyaml`: Leitura dos arquivos de configuração.
*   `pytest`: Execução da suíte de testes de integridade.

---

## 🧪 Validação e Testes

Para garantir que a migração para a nuvem não corrompa a integridade dos dados, implementamos uma suíte de testes automáticos que valida a estrutura dos arquivos Parquet gerados.

### O que é validado:
- **Shape e Colunas**: Verifica se o arquivo contém as 810 colunas (Opção B - 200 níveis).
- **Ordenação do Book**: Garante que Bids estão em ordem decrescente e Asks em crescente.
- **Spread Positivo**: Valida que o melhor Bid é sempre menor que o melhor Ask (sem book cruzado).
- **Qualidade das Features**: Certifica-se de que não existem NaNs ou Infs nas 9 features de treinamento.
- **Continuidade Temporal**: Verifica se os dados estão em ordem cronológica e sem gaps inesperados.

### Como rodar os testes:
```bash
pytest tests/test_cloud_etl_output.py
```
