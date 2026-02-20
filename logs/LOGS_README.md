# 📁 logs/ — Logging Directory

Esta pasta centraliza **todos os logs em tempo de execução** do pipeline QuantGod Cloud.
Cada subpasta corresponde a uma etapa distinta do sistema e é populada automaticamente
ao executar os scripts correspondentes.

> **Importante:** Nenhum arquivo de log é versionado pelo git (exceto `.gitkeep` para preservar a estrutura de diretórios). Adicione `logs/**/*.log` ao `.gitignore` caso ainda não esteja.

---

## 📂 Estrutura de Subpastas

```
logs/
├── debug_plots/        ← Gráficos de diagnóstico (ETL, features, distribuições)
├── etl/                ← Logs do pipeline de extração, transformação e carga
├── labelling/          ← Logs da geração de labels SELL/NEUTRAL/BUY
├── optimization/       ← Logs de cada trial Optuna + best params encontrados
├── training/           ← Logs do treinamento do Hybrid_TCN_LSTM por epoch
└── .gitkeep            ← Mantém a estrutura de diretórios no repositório
```

---

## 📁 `etl/`

**Gerado por:** `src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py`

**Conteúdo:** Um arquivo por execução do pipeline ETL, nomeado com timestamp:
```
etl/etl_YYYYMMDD_HHMMSS.log
```

**O que está registrado:**
- Arquivos ZIP encontrados no GDrive (via rclone)
- Snapshot inicial do orderbook recebido para cada arquivo
- Número de linhas geradas após resample de 1 minuto
- Valores de NaN/Infinity detectados pelo `DataValidator`
- Gaps temporais significativos no orderbook
- Caminho do arquivo Parquet salvo e tamanho final
- Erros de processamento por arquivo (continua nos próximos sem parar)

**Formato de exemplo:**
```
2026-02-18 10:32:01 [INFO]  Processing: 2025-03-14_BTCUSDT_ob200.data.zip
2026-02-18 10:32:03 [INFO]  Rows after resample: 1440
2026-02-18 10:32:03 [INFO]  Saved: data/L2/pre_processed/2025-03-14.parquet (2.1 MB)
```

---

## 📁 `labelling/`

**Gerado por:** `src/cloud/base_model/labelling/run_labelling.py`

**Conteúdo:**
```
labelling/labelling_processing.log           ← Log acumulativo de todas as execuções
labelling/labelling_SUFIXO_YYYYMMDD_HHMMSS.log  ← Log por configuração específica
```

> O sufixo do arquivo é derivado automaticamente do `labelled_dir` configurado em `labelling_config.yaml` (ex: `SELL_0004_BUY_0008_1h`). Isso garante que rodar com configurações diferentes nunca sobrescreva logs anteriores.

**O que está registrado:**
- Configuração usada: `lookahead`, `threshold_long`, `threshold_short`
- Arquivos processados em paralelo (`ProcessPoolExecutor`)
- Distribuição de labels por arquivo (SELL / NEUTRAL / BUY em %)
- Distribuição final agregada de todo o dataset
- Erros de processamento isolados por arquivo

**Formato de exemplo:**
```
2026-02-19 14:21:05 [INFO]  Config: lookahead=60, short=-0.004, long=0.008
2026-02-19 14:21:06 [INFO]  2025-03-14.parquet → SELL: 12.3% | NEU: 74.1% | BUY: 13.6%
2026-02-19 14:22:30 [INFO]  FINAL DISTRIBUTION → SELL: 11.8% | NEU: 75.2% | BUY: 13.0%
```

---

## 📁 `optimization/`

**Gerado por:** `src/cloud/base_model/otimizacao/run_optuna.py`

**Conteúdo:**
```
optimization/optimization_SUFIXO_YYYYMMDD_HHMMSS.log
```

> O sufixo é derivado do `labelled_dir` — o mesmo mecanismo do labelling — para rastreabilidade entre configuração de dados e resultado da otimização.

**O que está registrado:**
- Parâmetros testados em cada trial (`tcn_channels`, `lstm_hidden`, `lr`, `dropout`, `seq_len`, etc.)
- F1 Macro e F1 por classe (SELL/NEUTRAL/BUY) por epoch de cada trial
- LR atual a cada epoch (CosineAnnealingLR)
- Trials pruned pelo `MedianPruner` ou por OOM (CUDA out of memory)
- Melhor trial ao final e seus parâmetros

**Formato de exemplo:**
```
2026-02-20 09:14:00 [INFO]  Trial 3 START | tcn=64, lstm=256, seq=720, lr=0.000312
2026-02-20 09:16:40 [INFO]  Trial 3, Epoch 3/5 | F1 Macro: 0.4821 | F1 [S/N/B]: [0.441/0.523/0.483]
2026-02-20 09:18:00 [WARNING] Trial 7 — CUDA OOM! Clearing cache and pruning...
2026-02-20 09:45:00 [INFO]  Best F1 Macro: 0.5103 | Params: {tcn_channels: 64, lstm_hidden: 256, ...}
```

---

## 📁 `training/`

**Gerado por:** `src/cloud/base_model/treino/run_training.py`

**Conteúdo:**
```
training/training_YYYYMMDD_HHMMSS.log
```

**O que está registrado:**
- Device utilizado (CUDA / CPU) e número de parâmetros do modelo
- Class weights carregados de `base_model_config.yaml`
- Progresso por batch (a cada 200 batches): loss atual e % do epoch
- Por epoch completo:
  - Train Loss e Val Loss médios
  - F1 Macro e F1 Weighted
  - F1 por classe: `[SELL / NEUTRAL / BUY]`
  - Learning Rate atual
- Checkpoint salvo quando F1 Macro melhora
- Early stopping ativado (com o epoch em que ocorreu)

**Formato de exemplo:**
```
2026-02-20 11:00:00 [INFO]  Device: cuda | Parameters: 1,842,819
2026-02-20 11:00:00 [INFO]  CrossEntropyLoss weights: [2.0, 1.0, 2.0]
2026-02-20 11:03:22 [INFO]  Epoch 1/10 | Train Loss: 0.8841 | Val Loss: 0.9102
                             F1 Macro: 0.4103 | F1 [S/N/B]: [0.381/0.452/0.398] | LR: 0.000300
2026-02-20 11:03:22 [INFO]  ✅ Best model saved (F1 Macro: 0.4103)
```

---

## 📁 `debug_plots/`

**Gerado por:** scripts de diagnóstico e análise exploratória (uso manual).

**Conteúdo:** Imagens `.png` geradas durante análises de:
- Distribuição de features (histogramas pré/pós normalização)
- Séries temporais de micro_price e features derivadas
- Matrizes de correlação de features
- Análise de distribuição de labels por período

> Esta pasta é de uso **exclusivamente manual/diagnóstico**. Não é populada pelo pipeline automatizado.

---

## ⚙️ Boas Práticas

### Retenção de Logs
- **ETL e Labelling:** manter os últimos 30 arquivos (1 por dia processado)
- **Optimization:** manter todos — cada arquivo documenta um experimento único
- **Training:** manter todos — cada arquivo corresponde a um modelo salvo específico

### Naming Convention
Todos os arquivos seguem o padrão:
```
{modulo}_{sufixo_config}_{YYYYMMDD_HHMMSS}.log
```
O sufixo de configuração previne sobrescrita ao rodar experimentos com parâmetros diferentes.

### Limpeza Manual (RunPod)
Ao encerrar um Pod, comprimir os logs antes de desligar:
```bash
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
rclone copy logs_backup_*.tar.gz drive:QuantGod/logs_backup/
```

### Rotação de Logs (Opcional)
Para execuções contínuas de longa duração, configurar rotação via `logging.handlers.RotatingFileHandler`:
```python
handler = RotatingFileHandler(log_path, maxBytes=50*1024*1024, backupCount=5)
# Máximo 50 MB por arquivo, mantém 5 backups
```
