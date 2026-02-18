# QuantGod Data Directory 📊

Este diretório gerencia o ciclo de vida dos dados do projeto, desde a ingestão bruta até as features processadas para os modelos de Deep Learning.

## 📂 Estrutura de Pastas

### 1. `L2/`
Dados de Level 2 (Order Book).
*   **`raw/`**: Armazena os arquivos brutos (ZIP/JSON) provenientes de fontes externas (LakeAPI, Bybit, etc.).
    *   `l2_samples/`: Amostras reduzidas para desenvolvimento e testes rápidos.
*   **`pre_processed/`**: Dados em estágio intermediário de processamento (ex: reconstrução do book e amostragem temporal).
    *   *Validação via:* `pytest tests/test_preprocessed_quality.py`
*   **`labelled/`**: Dataset com as labels de target aplicadas (ex: buy/sell signals baseados em janelas futuras).

### 2. `processed/` (Feature Store)
Dataset final pronto para o modelo.
*   Contém arquivos **.parquet** otimizados.
*   Dados com feature engineering aplicada (Micro-price, Spread, OBI, etc.).
*   Normalização e stationarity fix aplicados.

### 3. `live/`
Dados de execução em tempo real.
*   Snapshots capturados via WebSocket durante a execução do bot.
*   Base para predições em real-time.

### 4. `artifacts/`
Objetos auxiliares do pipeline de dados.
*   **Scalers**: Arquivos `.pkl` com parâmetros de normalização (StandardScaler).
*   **Metadata**: Arquivos JSON/CSV de auditoria e logs de integridade do dataset.

---

## ✅ Qualidade e Integridade de Dados
Para garantir a robustez institucional, todos os dados em `data/L2/pre_processed` devem passar pelo teste de qualidade:
```powershell
pytest tests/test_preprocessed_quality.py
```
**O que é validado:**
- **Continuidade**: Nenhuma lacuna de dias no histórico.
- **Densidade**: 1440 linhas (amostras de 1 min) por arquivo diário.
- **Esquema**: Presença de todas as features de Candle Shape e Orderbook.
- **Qualidade**: Zero valores nulos (NaNs) nas features críticas.
- **Monotonicidade**: Timestamps estritamente crescentes sem duplicatas.

---

## ⚠️ Observações de Git
A maioria dos arquivos nestas pastas é ignorada pelo Git (`.gitignore`) devido ao tamanho. Para recriar os diretórios em um novo ambiente, utilize o script de setup adequado.
