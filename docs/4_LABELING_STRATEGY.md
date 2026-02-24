# 🏷️ 5. Labeling Strategy

> **Target Audience**: Traders, Quants, Risk Managers.
> **Script:** `src/cloud/base_model/labelling/run_labelling.py`
> **Config:** `src/cloud/base_model/labelling/labelling_config.yaml`

---

## 🎯 Filosofia — Asymmetric Threshold Labeling

O QuantGod usa **Thresholds Assimétricos por design**. Não prevemos simplesmente "sobe" ou "cai" — prevemos **resultados econômicos assimétricos**:

- Uma **compra** exige retorno esperado **2× maior** que uma venda.
- Isso reflete a assimetria real do mercado de futuros de BTC: o risco de liquidação de uma posição vendida em alta volatilidade é maior que o de uma posição comprada, exigindo prêmio de risco maior para justificar a entrada longa.

---

## 📐 Lógica de Rotulagem

Para cada candle `t`, calcula-se o **retorno cumulativo futuro** sobre uma janela de `lookahead` minutos:

```python
# Polars — run_labelling.py
future_return = pl.col("log_ret_close") \
    .rolling_sum(window_size=lookahead) \
    .shift(-lookahead)   # Alinha o retorno futuro ao instante t

# Aplicação dos thresholds
target = when(future_return > threshold_long).then(2)   # BUY
       .when(future_return < threshold_short).then(0)   # SELL
       .otherwise(1)                                     # NEUTRAL
```

> **Zero leakage:** o shift de `-lookahead` garante que o label no instante `t` usa apenas preços de `t+1` a `t+lookahead` — sem dados do futuro além do janela definida.

---

## 📊 Classes do Target

| Valor | Classe | Condição | Notas |
|:---:|:---|:---|:---|
| `0` | **SELL** | `future_return < threshold_short` | Movimento descendente significativo |
| `1` | **NEUTRAL** | `threshold_short ≤ return ≤ threshold_long` | Ruído → não negociar |
| `2` | **BUY** | `future_return > threshold_long` | Movimento ascendente forte |

---

## ⚙️ Configuração Ativa

O experimento ativo é `labelled_SELL_0004_BUY_0008_1h`:

| Parâmetro | Valor | Descrição |
|:---|:---|:---|
| `lookahead` | 60 candles | Janela de 60 minutos à frente |
| `threshold_short` | `-0.004` (~-0.4%) | Retorno < -0.4% em 60min → SELL |
| `threshold_long` | `+0.008` (~+0.8%) | Retorno > +0.8% em 60min → BUY |
| `input_dir` | `data/L2/pre_processed/` | |
| `output_dir` | `data/L2/labelled_SELL_0004_BUY_0008_1h/` | |

**Por que +0.8% e não +0.4% para BUY?**  
O threshold assimétrico (+0.8% vs -0.4%) filtra ruído e força o modelo a aprender apenas oportunidades de compra com sinal forte, evitando falsos positivos em sideways markets.

---

## 🧪 Experimentos de Labelling Disponíveis

Oito configurações foram testadas e seus datasets estão em `data/L2/`:

| Pasta | `threshold_short` | `threshold_long` | `lookahead` | Tamanho |
|:---|:---:|:---:|:---:|:---:|
| `labelled_SELL_0003_BUY_0005_1h` | -0.3% | +0.5% | 60 min | ~3.8 GB |
| `labelled_SELL_0004_BUY_0004_1h` | -0.4% | +0.4% | 60 min | ~3.8 GB |
| `labelled_SELL_0004_BUY_0004_2h` | -0.4% | +0.4% | 120 min | ~3.6 GB |
| `labelled_SELL_0004_BUY_0005_1h` | -0.4% | +0.5% | 60 min | ~3.8 GB |
| `labelled_SELL_0004_BUY_0006_1h` | -0.4% | +0.6% | 60 min | ~3.8 GB |
| **`labelled_SELL_0004_BUY_0008_1h`** | **-0.4%** | **+0.8%** | **60 min** | **~3.8 GB** | **← ativo** |
| `labelled_SELL_0004_BUY_0008_2h` | -0.4% | +0.8% | 120 min | ~3.6 GB |
| `labelled_SELL_0004_BUY_001_2h` | -0.4% | +1.0% | 120 min | ~3.6 GB |

> Para trocar de experimento, atualize `labelled_dir` em ambos `training_config.yaml` e `auditor_config.yaml`. O teste `test_config_integrity.py::TestCrossConfigConsistency` verifica que os dois arquivos apontam para o mesmo diretório.

---

## 🔧 Como Gerar um Novo Experimento

```bash
# 1. Copiar o config padrão
cp src/cloud/base_model/labelling/labelling_config.yaml \
   src/cloud/base_model/labelling/labelling_config_custom.yaml

# 2. Editar thresholds e output_dir
# labelling_config_custom.yaml:
# paths:
#   input_dir: "data/L2/pre_processed"
#   output_dir: "data/L2/labelled_SELL_0005_BUY_001_1h"
# params:
#   lookahead: 60
#   threshold_long: 0.010
#   threshold_short: -0.005

# 3. Rodar com config customizado
python src/cloud/base_model/labelling/run_labelling.py \
  src/cloud/base_model/labelling/labelling_config_custom.yaml

# 4. Validar
pytest tests/test_labelling_output.py \
  --labelled-dir data/L2/labelled_SELL_0005_BUY_001_1h -v
```

---

## ⚖️ Class Weights (Compensação de Desbalanceamento)

A classe NEUTRAL domina o dataset (~60-80% dependendo dos thresholds). Para compensar:

```yaml
# base_model_config.yaml
training:
  class_weights: [2.0, 1.0, 2.0]   # SELL / NEUTRAL / BUY
```

`CrossEntropyLoss(weight=[2.0, 1.0, 2.0])` penaliza erros em SELL e BUY com **2× mais peso** que erros em NEUTRAL.

---

## ✅ Validação do Output

```bash
# Verifica schema (target presente, int, sem nulls), valores {0,1,2},
# ≥ 2 classes por arquivo, ≥ 3% SELL e ≥ 3% BUY globalmente
pytest tests/test_labelling_output.py -v
```

**Sinais de alerta:**
- Se SELL ou BUY < 3%: threshold muito agressivo (a maioria dos samples cai em NEUTRAL)
- Se uma classe ausente completamente: bug no threshold ou dataset com pouca variação
- Se row count < 1380: lookahead trim anômalo (esperado: 1440 - lookahead = 1380 mín.)

---

## 📝 Output Schema

Arquivos em `data/L2/labelled_*/YYYY-MM-DD_BTCUSDT_ob*.parquet`:
- **Todas as 833 colunas** do `pre_processed/` +
- Coluna `target` (int) com valores `{0, 1, 2}` +
- Sem coluna `future_return` (removida após labelling)
- **Linhas**: `len(pre_processed) - lookahead` (últimas `lookahead` linhas são removidas pois não têm retorno futuro)
