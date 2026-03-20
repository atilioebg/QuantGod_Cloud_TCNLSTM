# 🏷️ 4. Labeling Strategy (Triple Barrier Prado-IID)

> **Target Audience**: Traders, Quants, Risk Managers.
> **Script:** `src/cloud/base_model/labelling/run_labelling.py`
> **Config:** `src/cloud/base_model/configs/master_config.yaml` (Secção `pre_processing.labelling`)

---

## 🎯 Filosofia — Tripla Barreira Direcional (Path-Dependent)

O QuantGod v5.0 utiliza o método de **Triple Barrier Labeling** proposto por Marcos López de Prado. Diferente da rotulagem simples (ponto-a-ponto), este método considera o **caminho** que o preço percorreu. Não importa apenas onde o preço está daqui a 15 minutos, mas se ele tocou seu Stop Loss ou Take Profit **antes** desse tempo acabar.

---

## 📐 Lógica das Três Barreiras

Para cada barra gerada pelo ETL, o motor de rotulagem projeta três limites no futuro:

1.  **Barreira Superior (Horizontal/TP):** Definida por $2.0 \times \sigma_{ewma}$ (volatilidade adaptativa). Representa o sucesso da operação (BUY se tocada primeiro).
2.  **Barreira Inferior (Horizontal/SL):** Definida por $1.0 \times \sigma_{ewma}$. Representa a falha da tese (SELL se tocada primeiro).
3.  **Barreira Temporal (Vertical/Time-Stop):** Definida por um tempo fixo (ex: 15 minutos). Representa a exaustão da janela de oportunidade (NEUTRAL se atingida).

---

## 📈 Volatilidade Adaptativa (EWMA)

O sistema não usa thresholds fixos (ex: 0.5%). Em vez disso, ele "escuta" o coração do mercado:
*   Calculamos a **EWMA Volatility** (Exponentially Weighted Moving Average) dos log-retornos das últimas 100 barras.
*   **Em Alta Volatilidade:** As barreiras se expandem (evita stop-out por ruído).
*   **Em Baixa Volatilidade:** As barreiras se contraem (captura movimentos curtos).

---

## ⚖️ Classes do Target

| Valor | Classe | Condição de Fechamento | Significado para o Modelo |
|:---:|:---|:---|:---|
| `0` | **SELL** | Tocou a barreira inferior primeiro | Tendência de queda limpa detectada |
| `1` | **NEUTRAL** | Atingiu a barreira vertical | Ruído ou falta de momentum claro |
| `2` | **BUY** | Tocou a barreira superior primeiro | Tendência de alta limpa detectada |

---

## 📈 Specialist Training (K-Fold OOF)
Para evitar que o modelo aprenda apenas o "barulho" de curtos períodos, o sistema agora utiliza:
1.  **K-Fold Specialist**: O dataset de 345M é dividido cronologicamente em K-folds.
2.  **OOF Meta-Features**: O modelo gera predições "Out-of-Fold" (sem leakage) que servem de entrada para o Auditor XGBoost.
3.  **Sniper Score Optimization**: A métrica de sucesso foca 70% na acurácia direcional (Buy/Sell) e 30% na global (F1-Macro).

---

## ⚖️ First Touch Rule (SOTA AFML)

Se dentro da janela de 15 minutos o preço for tão volátil que atinja **tanto o Take Profit quanto o Stop Loss**, o motor utiliza a regra do **Primeiro Toque** (`use_first_touch: true`):
*   O sistema captura o índice temporal exato de cada toque.
*   **Decisão**: O evento que ocorreu **primeiro** define o label. 
*   **Vantagem**: Recupera amostras que antes seriam descartadas como "ambíguas", aumentando a densidade do dataset sem perder a integridade estatística.

---

## ⚙️ Configuração Ativa

Parâmetros gerenciados centralmente no `master_config.yaml`:

| Parâmetro | Valor Padrão | Descrição |
|:---|:---|:---|
| `horizon_minutes` | 15 min | Limite máximo de tempo para a barreira vertical |
| `use_first_touch` | true | Ativa resolução temporal de ambiguidades (AFML SOTA) |
| `pt_multiplier` | 2.0 | Multiplicador da volatilidade para o Take Profit |
| `sl_multiplier` | 1.0 | Multiplicador da volatilidade para o Stop Loss |
| `vol_span` | 100 bars | Janela de lookback para o cálculo da EWMA |

---

## 🔬 Validação de Qualidade

```bash
# Verifica distribuição das classes, integridade cronológica e ausência de leakage
pytest tests/labelling/test_labelling_output.py -v
```

**Sinais de Alerta:**
*   Se o percentual de BUY/SELL cair abaixo de 2%: Multiplicadores muito altos ou mercado muito lateral.
*   Explosão de amostras Neutras: Janela vertical muito curta ou volatilidade subestimada. Use o **Subsampling Probabilístico** (`subsample_keep_ratio: 0.2`) para reduzir o peso estatístico do ruído sem perder a cronologia.

---

Consulte [`7_DATA_REFERENCE.md`](7_DATA_REFERENCE.md) para detalhes sobre as features de entrada.
