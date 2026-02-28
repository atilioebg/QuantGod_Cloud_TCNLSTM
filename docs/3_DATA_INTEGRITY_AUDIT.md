# QuantGod v4.6: Filosofia de Integridade e Auditoria "Gold Standard"

Este documento detalha o motor de **Data Quality Assurance** do projeto QuantGod v4.6 Gold, distinguindo falhas técnicas de anomalias reais de mercado.

## 1. Erro Técnico vs. Anomalia de Mercado

O pipeline v4.6 distingue corrupção física de dados de eventos de volatilidade real.

### A. Erros Técnicos: Rejeição Total (Hard Skip)
- **Gaps Temporais Críticos**: Ausência de dados > 60 minutos após tentativa de cura. (Threshold v4.6 Gold).
- **Preço Estático (Stale Data)**: Zero variação de preço/volume (feed travado).
- **Inconsistência de Shape**: Quantidade de colunas divergente do `master_config.yaml`.
- **Ghost Features**: Detecção de sinais/logits não autorizados no config.
- **NaNs/Infs**: Nulos/Infinitos em campos vitais.

## 2. Tratativa de Nível 1: Cura de Dados (Healing)
Para gaps curtos (< 5 minutos), o sistema aplica a **Tratativa Tripla** para evitar o descarte desnecessário de dados:
1.  **Grupo A (Interpolação Linear):** Aplicada a colunas de OHLC e volatilidade para suavizar a transição de preços.
2.  **Grupo B (Median-Fill):** Aplicada a métricas de microestrutura (OBI, VPIN, RDI) usando a mediana do arquivo atual para neutralidade estatística.
3.  **Grupo C (Zero-Fill):** Aplicada a fluxos de eventos (OFI, tick_count, momentum) para sinalizar ausência de atividade.

---

## 3. Mapa de Artefatos Gerados (Auditoria Gold)

Ao final de cada etapa do pipeline, os seguintes arquivos são gerados para rastro permanente:

### A. Logs de Processamento
*   **Caminho**: `logs/etl/etl_YYYYMMDD_HHMMSS.log`
*   **Conteúdo**: Registro temporal de download, streaming, clipping e validação de integridade.

### B. Relatório Executivo de Auditoria (CSV)
*   **Caminho**: `docs/reports/audit_summary.csv`
*   **Conteúdo**: `status` (VALID/INVALID/FIXED), `max_gap_before`, `max_gap_after`, `features_healed` e `healing_details`.
*   **Uso**: Auditoria rápida de grandes volumes de arquivos.

### C. Relatório Detalhado de Sanidade (JSON)
*   **Caminho**: `docs/reports/data_quality_report.json`
*   **Conteúdo**: 
    - `lineage_summary`: Contagem de colunas por origem.
    - `high_tails_detail`: Registro de anomalias de cauda (Z-Score > 12).
    - `dead_features_lineage`: Mapeamento de features sem variância.

### D. Datasets Finais
*   **Local**: `data/L2/pre_processed_L2/*.parquet`
*   **Nuvem**: `drive:PROJETOS/PRE_PROCESSED_L2_V4.6_GOLD_1min_30F`

---

## 4. Estratégia de Soft Clipping (Winsorization)
O sistema calcula o **P99** e aplica um teto de `10x P99` para "domar" extremos de liquidez sem perder o sinal econômico.
