# QuantGod v4.6: Filosofia de Integridade e Auditoria "Gold Standard"

Este documento detalha o motor de **Data Quality Assurance** do projeto QuantGod v4.6 Gold, distinguindo falhas técnicas de anomalias reais de mercado.

## 1. Erro Técnico vs. Anomalia de Mercado

O pipeline v4.6 distingue corrupção física de dados de eventos de volatilidade real.

### A. Erros Técnicos: Rejeição Total (Hard Skip)
- **Gaps Temporais**: Ausência de dados > 25 minutos.
- **Preço Estático (Stale Data)**: Zero variação de preço/volume (feed travado).
- **Inconsistência de Shape**: Quantidade de colunas divergente do `master_config.yaml`.
- **Ghost Features**: Detecção de sinais/logits não autorizados no config.
- **NaNs/Infs**: Nulos/Infinitos em campos vitais.

### B. Linhagem de Dados (Feature Lineage)
Cada métrica é rotulada para auditoria profunda:
- **`[DNN_INPUT]`**: As 30 features que alimentam o "cérebro" TCN-LSTM.
- **`[XGB_ONLY]`**: Sinais e Probabilidades cruzadas usadas apenas pelo Juiz Auditor.
- **`[RAW_DATA]`**: Dados brutos do Order Book (L2) preservados para suporte mas ignorados pelo modelo.

---

## 2. Mapa de Artefatos Gerados (Auditoria Gold)

Ao final de cada etapa do pipeline, os seguintes arquivos são gerados para rastro permanente:

### A. Logs de Processamento
*   **Caminho**: `logs/etl/etl_YYYYMMDD_HHMMSS.log`
*   **Conteúdo**: Registro temporal de download, streaming, clipping e validação de integridade.
*   **Uso**: Debug operacional e verificação de fluxo.

### B. Relatório de Métricas de Sanidade (JSON)
*   **Caminho**: `docs/reports/data_quality_report.json`
*   **Conteúdo**: 
    - `lineage_summary`: Contagem de colunas por origem.
    - `high_tails_detail`: Registro de anomalias de cauda (Z-Score > 12).
    - `dead_features_lineage`: Mapeamento de features sem variância.
    - `is_valid`: Selo final de qualidade.

### C. Datasets Finais
*   **Local**: `data/L2/pre_processed_L2/*.parquet`
*   **Nuvem**: `drive:PROJETOS/PRE_PROCESSED_L2_V4.6_GOLD_1min_30F`

---

## 3. Estratégia de Soft Clipping (Winsorization)
O sistema calcula o **P99** e aplica um teto de `10x P99` para "domar" extremos de liquidez sem perder o sinal econômico. Este processo é totalmente registrado no JSON de auditoria acima.
