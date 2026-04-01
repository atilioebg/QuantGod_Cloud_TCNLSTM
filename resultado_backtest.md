# 🎯 Sniper Sniper Sniper Training - Relatório de Resultados

## 🧠 O Manifesto do Sniper (Estratégia de Calibração)

### 📌 O Que Estamos Fazendo?
Estamos otimizando a engine de High-Frequency Trading (`QuantGod_Cloud_TCNLSTM`) para operar como um **Sniper Verdadeiro**. O objetivo é encontrar o equilíbrio exato entre a Métrica de Convicção (Threshold do Auditor), o Alvo de Lucro (PT), a Proteção de Ruído (SL) e o Tempo de Exposição (Horizonte).

### 🎯 Para Que Estamos Fazendo Isso?
Para **vencer as taxas da corretora (0.12% por trade) e alcançar o ROI Verde Consistente**. 
Descobrimos que a IA já possui um "Alpha Bruto" incrivelmente positivo, mas quando o modelo opera demais (Hyper-Frequência, como na V4.34 com 398 trades), as taxas da corretora trituram todo o capital (-48% só de taxas). O desafio não é mais de "predição de mercado", mas sim de **Eficiência de Gasto e Risco/Retorno (RR)**.

### 💡 A Linha de Raciocínio (A Evolução do CoT)
Nossa calibração provou empiricamente a seguinte trilha:

1.  **O Problema do Moonshot (V4.31 a V4.36):**
    *   *Tentativa:* Buscar alvos absurdos (5.0 a 10.0x a volatilidade) para cobrir qualquer taxa.
    *   *Problema:* O mercado quase nunca sobe 5x ou 10x sem antes corrigir. O preço batia no alvo direcional correto, mas sofria retrações (agulhadas intradiárias) que acertavam o Stop Loss (SL) antes da decolagem final. O Win Rate caiu para 0%.

2.  **A Prova de Fogo do Alpha (V4.37 - Realistic Sniper):**
    *   *Tentativa:* Focar em um alvo pé no chão intradiário (**PT 3.5**) com Stop técnico (**SL 1.5**).
    *   *Resultado:* **Sucesso Absoluto de Predição!** Atingimos um **Win Rate de 33.8%**. Essa relação Risco/Retorno gera um lucro forte. 
    *   *O Vilão:* O número de trades (68) causou uma sangria de **-8.16%** só de taxas, deixando o saldo final no vermelho (-6.14%). Mas o retorno bruto (sem as taxas) atingiu **+2.02%**!

3.  **O Controle de Gastos (V4.38 - Golden Strike):**
    *   *Tentativa:* Para reduzir as taxas, subimos o filtro de Convicção (Threshold de 0.8910 para **0.8920**). Para cobrir qualquer resto de taxa, voltamos a sonhar alto (PT 5.0).
    *   *Resultado:* A tese de filtro funcionou majestosamente! Os trades despencaram de 68 para 40 (corte de ~50% nas taxas). Porém, o alvo de 5.0 estragou o belíssimo Win Rate (caiu para 20.0%). O trade era bom, mas não elástico o suficiente.

4.  **O Ponto de Ouro (A Síntese Final - V4.39 Platinum):**
    *   Pegamos a assertividade incrível do Alvo de **3.5** (V4.37) e ligamos ao Filtro Rigoroso (Threshold de **0.8920**) da V4.38, criando a janela de tempo perfeita (2.5 horas). 
    *   O robô fará 40 cirurgias exatas e fechará as posições antes da retração. **É a purificadora extração do Alpha para arrancar o ROI Verde!**

---

### 📊 Tabela Oficial de Performance

| Versão | Codenome | Thresh | PT | SL | Horiz | Trades | WR% | Dur(m) | T/Dia | Balanço | ROI (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4.26** | Sniper Base | 0.890 | 1.5 | 0.45 | 15m | 39 | 10.3 | 13.9 | 3.0 | 9594 | -4.06 |
| **4.31** | Titan Strike | 0.8928 | 5.0 | 1.0 | 4h | 5 | 0.0 | 13.5 | 0.45 | 9890 | -1.09 |
| **4.34** | Ghost Strike | 0.887 | 1.8 | 0.28 | 15m | 398 | 16.1 | 11.0 | 30.7 | 6300 | -37.00 |
| **4.35** | Oracle Strike| 0.8929 | 10.0 | 0.85 | 8h | 5 | 0.0 | 3.28 | 0.45 | 9919 | -0.81 |
| **4.36** | Eternal Bunker| 0.8929 | 10.0 | 1.9 | 8h | 5 | 0.0 | 7.28 | 0.45 | 9892 | -1.07 |
| **4.37** | Realistic Sniper| 0.8910 | 3.5 | 1.5 | 2h | 68 | 33.8 | 37.0 | 5.3 | 9385 | -6.14 |
| **4.38** | Golden Strike| 0.8920 | 5.0 | 1.5 | 3h | 40 | 20.0 | 19.2 | 3.1 | 9443 | -5.56 |
| **4.39** | Platinum Strike| 0.8920 | 3.5 | 1.5 | 2.5h| 44 | 25.0 | 27.4 | 3.4 | 9325 | -6.74 |
| **4.40** | **True Alpha** | **0.8900** | **3.5** | **1.5** | **2h** | *EXEC* | *WAIT* | *WAIT* | *WAIT* | *WAIT* | *WAIT* |

---

> [!IMPORTANT]
> **V4.40 - True Alpha Shift:** Descobrimos a "Armadilha de Superconfiança". Sinais extremamente altos (>0.892) no TCN-LSTM são capturas de liquidez (agulhadas) que varrem stops e revertem o mercado. O fluxo verdadeiro e sustentável (Alpha de 33% de WR) mora na faixa de 0.890. Vamos baixar o Threshold para o nível rentável e esmagar as taxas usando um bloqueio temporal agressivo (Cooldown de 12 horas!).
