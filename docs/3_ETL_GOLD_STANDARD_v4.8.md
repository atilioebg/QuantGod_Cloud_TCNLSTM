# ETL Gold Standard: Selo de Aprovação v4.8

Este documento detalha o que significa um arquivo ser aprovado pelo pipeline `run_pipeline.py` e pelos testes de integridade `run_quality_tests.py`. A aprovação garante que o dado passou por **4 Camadas de Auditoria Institucional**.

## 1. Integridade de Arquitetura (O "Shape")
Garante que o arquivo possui a estrutura correta para os modelos TCN/XGBoost.
- **Checklist de Features:** Verificação das 30 features Snipers e Institucionais (ex: `kyle_lambda`, `vpin_min25`, `book_asymmetry_v5`).
- **Níveis do Orderbook:** Presença obrigatória dos 200 níveis (P e S) para auditoria e reconstrução.
- **Deduplicação:** Garantia de que não existem colunas redundantes ou lixo de processamento.

## 2. Sanidade de Dados (Filtros Fatais)
Aplica critérios rigorosos de limpeza para evitar o treinamento com dados corrompidos.
- **Zero NaNs/Infs:** Arquivos com valores nulos ou infinitos são rejeitados imediatamente.
- **Monotonicidade Temporal:** Os timestamps devem ser estritamente crescentes. Inversões temporais causam rejeição fatal.
- **Clipping de Outliers:** Valores extremos que excedem 10x o P99 são "podados", protegendo os gradientes do modelo contra picos de volatilidade ruidosos.

## 3. Protocolo Island Split (Sobrevivência)
Garante que o dado possui contexto temporal suficiente para modelos de série temporal.
- **Regra dos 120 Minutos:** O arquivo deve conter pelo menos uma "ilha" de dados contínuos com duração superior a 2 horas.
- **Isolamento de Estado:** Impede que o estado de indicadores (como médias móveis) "vaze" entre gaps temporais através de forward-fills cegos.
- **Rejeição de Fragmentos:** Segmentos curtos (<120min) são descartados automaticamente por falta de contexto histórico.

## 4. Auditoria de Gaps (v4.8 Gold Healing)
Nível máximo de certificação para garantir continuidade perfeita.
- **Cura de Gaps Curtos (<= 5 min):** Cicatrização via interpolação linear (preços) e preenchimento zero (fluxos).
- **Sensibilidade de Borda:** Detecção de gaps no início (00:00) e fim (23:59) do dia para auditoria de completude total do arquivo.
- **Ghost Gap Prevention:** Garantia de que o relatório final de gaps (`max_gap_after`) seja `0.0` para arquivos curados, eliminando incertezas estatísticas.

---

### Status do Auditor (`audit_summary.csv`)
- **VALID:** O arquivo está íntegro e pronto para treinamento e backtest.
- **FIXED:** O arquivo possuía irregularidades menores que foram corrigidas automaticamente.
- **INVALID:** O arquivo foi descartado por falha grave de integridade ou falta de massa crítica (ilhas curtas).
