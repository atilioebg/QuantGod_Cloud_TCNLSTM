# 📡 2. Data Collection (Dados Brutos)

> **Target Audience**: Data Engineers.
> **Status:** Os dados históricos estão coletados e armazenados no Google Drive. Esta seção documenta como foram obtidos e como acessá-los.

---

## 📦 Fonte dos Dados

| Atributo | Valor |
|:---|:---|
| **Exchange** | Bybit Futures (histórico) / Binance Futures (live) |
| **Par** | BTC/USDT Perpetual Futures |
| **Tipo** | Level 2 Order Book — Profundidade Completa |
| **Formato** | Arquivos `.zip` contendo mensagens JSON |
| **Período** | 2023-01-01 a 2026-02-xx |
| **Localização** | Google Drive: `drive:PROJETOS/BTC_USDT_L2_2023_2026/` |
| **Tamanho total** | ~35.7 GB processado (`data/L2/pre_processed/`) |

---

## 📂 Organização dos Dados Brutos no Google Drive

```
drive:PROJETOS/BTC_USDT_L2_2023_2026/
├── 2023/
│   ├── 2023-01-01_BTCUSDT_ob500.data.zip
│   ├── 2023-01-02_BTCUSDT_ob500.data.zip
│   └── ...  (365 arquivos ob500 — 500 níveis de profundidade)
├── 2024/
│   ├── 2024-01-01_BTCUSDT_ob200.data.zip
│   └── ...  (366 arquivos ob200 — 200 níveis)
├── 2025/
│   └── ...  (ob200)
└── 2026/
    └── ...  (ob200, até data atual)
```

### Mudança de Profundidade em 2024

| Período | Profundidade | Arquivo |
|:---|:---|:---|
| 2023 | OB500 | `*_ob500.data.zip` |
| 2024–2026 | OB200 | `*_ob200.data.zip` |

> O ETL aplica **Hard Cut automático para 200 níveis** nos arquivos OB500, garantindo schema idêntico para todos os anos.

---

## 📋 Estrutura de Cada ZIP

Cada ZIP contém um único arquivo `.data` com sequência de mensagens JSON (uma por linha):

### Mensagem `snapshot` (Estado Inicial)
```json
{
  "type": "snapshot",
  "ts": 1704067200000,
  "data": {
    "b": [["43100.5", "1.234"], ["43100.0", "0.890"]],
    "a": [["43101.0", "2.100"], ["43101.5", "0.456"]]
  }
}
```

### Mensagem `delta` (Atualização Incremental)
```json
{
  "type": "delta",
  "ts": 1704067200150,
  "data": {
    "b": [["43100.5", "0.000"], ["43099.0", "5.000"]],
    "a": [["43102.0", "1.500"]]
  }
}
```

> **⚠️ Tamanho `"0.000"` em delta = remoção do nível de preço** (não é um nível com liquidez zero).

---

## 🔌 Acessar os Dados

Os dados ficam no Google Drive e são acessados via `rclone mount`:

```bash
# Linux/RunPod — montar em background
rclone mount drive: /workspace/gdrive --vfs-cache-mode full --allow-other &

# Verificar acesso
ls /workspace/gdrive/PROJETOS/BTC_USDT_L2_2023_2026/2024/ | head -5
```

Para download direto ao NVMe (acesso mais rápido durante treino):
```bash
# Download do dataset labelled ativo para disco local
tmux new -s download
mkdir -p /workspace/data/L2/labelled_SELL_0004_BUY_0008_1h
rclone copy drive:PROJETOS/L2/labelled_SELL_0004_BUY_0008_1h \
  /workspace/data/L2/labelled_SELL_0004_BUY_0008_1h -P
```

Consulte `src/cloud/README.md` → Seção **Guia Completo RunPod** para o fluxo detalhado.

---

## 🔄 Pipeline de Processamento

Os dados brutos não são usados diretamente pelo modelo. O fluxo completo é:

```
ZIPs (GDrive/Bybit)  →  ETL (transform.py)  →  pre_processed/*.parquet
                                                          ↓
                                                labelling (run_labelling.py)
                                                          ↓
                                              labelled_*/*.parquet + coluna target
                                                          ↓
                                                Treino / Optuna / XGBoost
```

Para detalhes de cada etapa, veja:
- **ETL:** [`3_DATA_ENGINEERING.md`](3_DATA_ENGINEERING.md)
- **Labelling:** [`5_LABELING_STRATEGY.md`](5_LABELING_STRATEGY.md)
- **Pipeline completo:** [`src/cloud/README.md`](../src/cloud/README.md)

---

## 🔴 Dados Live (Inferência em Produção)

Durante inferência ao vivo, o sistema **não usa os ZIPs do GDrive**. Em vez disso, o `binance_adapter.py` conecta via WebSocket ao Binance Futures e reconstrói o orderbook em tempo real, aplicando a **mesma lógica** de feature engineering do ETL histórico.

| Aspecto | Treinamento (Bybit) | Live (Binance) |
|:---|:---|:---|
| Fonte | ZIPs históricos `.data` | WebSocket `btcusdt@depth@100ms` |
| Snapshot inicial | `"type":"snapshot"` no arquivo | REST GET `/fapi/v1/depth?limit=1000` |
| Sync | Sequencial por arquivo | `lastUpdateId/U/u` + re-bootstrap |
| Features geradas | Idênticas (9 features) | Idênticas (9 features) |
| Scaler | `StandardScaler.fit()` no train set | `scaler_finetuning.pkl` carregado |

Veja [`docs/TCN_LSTM.md`](TCN_LSTM.md) → Seção 7 para a documentação completa do `binance_adapter.py`.
