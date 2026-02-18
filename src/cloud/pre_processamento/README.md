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

## 🚀 Como Usar

### Passo 1: Preparar a máquina
```bash
cd cloud
chmod +x setup_cloud.sh
./setup_cloud.sh
```

### Passo 2: Configurar e Ativar o Rclone
O processo de ativação depende do seu ambiente:

#### 1. No Windows (Seu PC atual)
Como o WinFSP já está instalado, você pode usar o `rclone.exe` presente na raiz do projeto para montar o Google Drive como um disco local (`Z:`):
1. Abra um terminal exclusivo para o rclone.
2. Execute o comando para montar:
   ```powershell
   .\rclone.exe mount drive: Z: --vfs-cache-mode full --config rclone.conf
   ```
3. **Mantenha o terminal aberto.** Se fechar, o disco `Z:` será desconectado.

#### 2. Na Cloud (Linux / RunPod)
No Linux, o mount é feito em um diretório do sistema:
1. Configure o acesso (caso ainda não tenha feito): `rclone config`.
2. Crie a pasta de destino: `mkdir -p /workspace/gdrive`.
3. Ative o mount em segundo plano:
   ```bash
   rclone mount drive: /workspace/gdrive --vfs-cache-mode full --allow-other &
   ```
   *O `&` no final libera o terminal para outros comandos.*

### Passo 3: Rodar o Processamento
Ative o ambiente e execute o pipeline:
```bash
source .venv/bin/activate
# Para produção (RunPod):
python -m src.cloud.pre_processamento.orchestration.run_pipeline
# Para testes (Local):
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
