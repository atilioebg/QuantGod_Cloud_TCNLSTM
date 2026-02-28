# QuantGod v4.5: Filosofia de Integridade e Auditoria "Gold Standard"

Este documento detalha o motor de **Data Quality Assurance** do projeto QuantGod, distinguindo falhas técnicas de anomalias reais de mercado (Microestrutura de Choque).

## 1. Erro Técnico (Bug) vs. Anomalia de Mercado (Insight)

O pipeline v4.5 distingue corrupção física de dados de eventos de volatilidade real.

### A. Erros Técnicos: Rejeição Total (Hard Skip)
Quando estas condições são detectadas, o arquivo é ignorado para preservar a sanidade do dataset:
- **Gaps Temporais**: Ausência de dados > 5 minutos em arquivos de 24h.
- **Preço Estático (Stale Data)**: Zero variação de preço/volume por janelas longas (indica feed travado).
- **Livro Cruzado (Crossed Book)**: Bid > Ask, indicando snapshot L2 corrompido na fonte.
- **NaNs/Infs**: Nulos/Infinitos em campos vitais.

### B. Anomalias de Mercado: Domesticação (Soft Clipping)
Estes são eventos **REAIS** na Binance, mas estatisticamente nocivos para o treinamento de modelos de IA (TCN/LSTM/XGBoost):
- **Microestrutura de Choque**: Momentos onde a liquidez some e ordens pequenas causam saltos gigantescos de preço (ex: `kyle_lambda` saltando de 0.5 para 1400.0 em 1 segundo).
- **Flash Crashes/Squeezes**: Movimentos extremos que destorcem a escala do Z-Score, "esmagando" a sensibilidade do modelo para movimentos normais.

---

## 2. Estratégia de Clipping (Winsorization)

**Racional**: Descartar 24h de dados úteis por causa de 1 minuto de caos é ineficiente. O Clipping mantém o contexto histórico "domando" os extremos.

### Lógica do Teto Dinâmico
1. O sistema calcula o **Percentil 99 (P99)** do arquivo atual.
2. Define o **Teto** como `P99 * 10` (Parametrizável via `master_config.yaml`).
3. Valores acima do teto são reduzidos para o limite superior antes da normalização.

---

## 3. Auditoria e Rastreabilidade

Ao final de cada processamento, o arquivo `docs/reports/data_quality_report.json` serve de certificado final:
- **`outlier_density`**: Prova quanto do dataset foi "domado".
- **`clipping_events`**: Detalha qual coluna foi corrigida e qual era o valor bizarro inicial.

Este processo garante que o dataset v4.5 seja o mais limpo e profissional possível para o fundo de investimento.
