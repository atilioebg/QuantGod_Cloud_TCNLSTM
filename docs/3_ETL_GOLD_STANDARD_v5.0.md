# ETL Gold Standard: Selo de Aprovação v9.5 (AFML Integrated)

Este documento detalha o que significa um arquivo ser aprovado pelo pipeline `run_pipeline.py` sob a arquitetura de **Domínio da Informação** alinhada ao Estado da Arte (SOTA) de Marcos López de Prado (AFML). A aprovação v9.5 garante que o dado é estatisticamente IID e livre de vazamentos temporais.

---

## 1. Integridade de Eventos (Cap. 2)
Garante que a reconstrução do fluxo de ordens foi perfeita e reativa ao mercado.
- **Conservação de Massa:** O volume total e a contagem de ticks das barras devem bater exatamente com os dados brutos (Reconciliação de Transbordo).
- **Amostragem IID:** Validação de que as barras (Dollar/Tick/OFI) recuperam a normalidade dos retornos.
- **CUSUM Feedback:** O filtro CUSUM deve disparar apenas em quebras de regime, com taxa de rejeição monitorada no `audit_report`.

---

## 2. Sanidade Microestrutural (Cap. 3)
Filtros aplicados para proteger o modelo contra anomalias técnicas e ruído.
- **CUSUM Adaptativo:** O limiar $h$ deve reagir à volatilidade EWMA local (`adaptive_h: true`), evitando sub-amostragem em períodos de baixa atividade.
- **First Touch Decision:** A rotulagem deve priorizar a ordem cronológica em casos de toques simultâneos em barreiras horizontais, eliminando o label 'Neutro' artificial por ambiguidade.
- **Stale Book Detection:** Marcação de barras onde o Orderbook não foi atualizado por mais de 2 segundos após um trade.

---

## 3. Protocolo de Segregação v9.5 (Cap. 7)
A continuidade agora é medida pelo **fluxo de informação** e a independência é garantida por silêncio estatístico.
- **Bar-level Split:** A divisão entre Treino e Validação deve respeitar a contagem exata de observações (`split_by_bars: true`), não apenas o número de arquivos.
- **Purge Gap:** Remoção obrigatória de $N$ minutos do final do treino para bloquear o lookahead do label.
- **Embargo Period:** Remoção de $N$ minutos do início da validação para eliminar a autocorrelação serial persistente.

---

## 4. Auditoria e Status (`audit_summary.json`)
O sistema agora gera um JSON estruturado com os seguintes selos:

- **🥇 GOLD (SOTA):** Arquivo 100% íntegro, amostragem adaptativa ativa, e gaps de purga/embargo validados via `Chrono Guard`.
- **🥈 SILVER:** Arquivo funcional, mas com uso de limiares estáticos ou gaps cicatrizados por interpolação.
- **❌ INVALID:** Descartado por violação cronológica, falta de massa crítica de eventos ou corrupção de timestamps.

---
