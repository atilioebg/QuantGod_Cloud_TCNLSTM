# 🧪 tests/ — Test Suite

Suite de testes automatizados do QuantGod Cloud, estruturada em três camadas:
**unitários** (sem dados/GPU), **integridade de config** (YAMLs), e **qualidade de dados** (Parquet).

---

## 📂 Estrutura

```
tests/
├── conftest.py                     ← Fixtures compartilhadas e constantes globais
├── test_model.py                   ← Testes unitários: Hybrid_TCN_LSTM
├── test_meta_features.py           ← Testes unitários: feature_engineering_meta.py
├── test_config_integrity.py        ← Validação dos 4 arquivos YAML de config
├── test_cloud_etl_output.py        ← Qualidade dos dados: pre_processed/ (pandas)
├── test_preprocessed_quality.py    ← Qualidade dos dados: pre_processed/ (polars + continuidade)
└── test_labelling_output.py        ← Qualidade dos dados: labelled_*/ (polars)
```

---

## ⚡ Execução Rápida

```bash
# ─── Usar sempre da raiz do projeto ───────────────────────────────────────────
cd C:\Users\Atilio\Desktop\PROJETOS\PESSOAL\QuantGod_Cloud   # ou /workspace

# Unitários + Config — sem GPU, sem dados, < 30 segundos
pytest tests/test_model.py tests/test_meta_features.py tests/test_config_integrity.py -v

# Config apenas (CI rápido)
pytest tests/test_config_integrity.py -v

# Qualidade de dados — requer data/L2/* populado
pytest tests/test_preprocessed_quality.py -v
pytest tests/test_labelling_output.py -v

# Tudo
pytest tests/ -v
```

### Trocar o experimento de labelling testado

```bash
# Testar um experimento específico sem editar código
pytest tests/test_labelling_output.py \
  --labelled-dir data/L2/labelled_SELL_0004_BUY_0006_1h -v
```

---

## 📋 Descrição dos Arquivos

### `conftest.py`

Fixtures e constantes compartilhadas automaticamente por todos os módulos via descoberta do pytest.

| Fixture / Constante | Tipo | Descrição |
|:---|:---|:---|
| `FEATURE_NAMES` | `list[str]` | 9 features do modelo — fonte única de verdade nos testes |
| `NUM_FEATURES` | `int` | 9 |
| `NUM_CLASSES` | `int` | 3 (SELL / NEUTRAL / BUY) |
| `SEQ_LEN` | `int` | 720 (lookback de 12h) |
| `META_FEATURES` | `int` | 14 (input do XGBoost) |
| `ACTIVE_LABELLED_DIR` | `Path` | Experimento de labelling ativo (`SELL_0004_BUY_0008_1h`) |
| `sample_micro_price` | fixture | Array (720,) sintético — micro_price series |
| `sample_probs_balanced` | fixture | Array (3,) somando 1.0 |
| `sample_last_step` | fixture | Array (9,) — último timestep sintético |
| `sample_sequence_batch` | fixture | Array (4, 720, 9) — batch mínimo |

---

### `test_model.py`

Testa `src/cloud/base_model/models/model.py` — **não requer GPU nem dados**.

> Módulo pulado automaticamente se `torch` não estiver instalado.

| Classe de Testes | O que valida |
|:---|:---|
| `TestModelInstantiation` | `Hybrid_TCN_LSTM` instancia com configs variadas; parâmetros treináveis > 0; count razoável (100K–20M) |
| `TestForwardPassShape` | Output é dict com chaves `logits` e `probs`; shapes `(B, 3)`; funciona com batch=1 e seq_len=1440 |
| `TestProbabilityInvariants` | `probs.sum(dim=-1) == 1.0`; todos ≥ 0; todos ≤ 1.0 |
| `TestCausalConv` | `CausalConv1d` preserva dimensão temporal; perturbação em `t > k` não altera output em `t ≤ k` |
| `TestDeterminism` | Em `eval mode`, mesmo input → mesmo output (sem Dropout estocástico) |
| `TestGradientFlow` | Todos os parâmetros recebem gradiente após `loss.backward()` |

**Input contratual:** `(B, seq_len, 9)` → `{"logits": (B, 3), "probs": (B, 3)}`

---

### `test_meta_features.py`

Testa `src/cloud/auditor_model/feature_engineering_meta.py` — **pure numpy, sem GPU**.

| Classe de Testes | O que valida |
|:---|:---|
| `TestExtractMetaFeatures` | Shape `(14,)`, dtype `float32`, sem NaN/Inf; probs passthrough; entropy ≥ 0; RSI em [0,100]; `last_step_features=None` → zeros |
| `TestRSI` | Bounded [0,100]; all-gains→100; all-losses→<10; flat→100 |
| `TestEMA` | Série plana → EMA = constante; EMA distance = 0 em série plana |
| `TestBollinger` | Janela plana → %B = 0.5 |
| `TestATR` | Não-negativo; zero em série plana |
| `TestEntropy` | Uniforme → máximo; certo → ~0; sempre ≥ 0 |
| `TestMetaFeatureNames` | 14 names, sem duplicatas, nomes obrigatórios presentes |

---

### `test_config_integrity.py`

Valida os 4 arquivos YAML de configuração — **sem dados, sem torch**.

| Classe de Testes | Config | O que valida |
|:---|:---|:---|
| `TestBaseModelConfig` | `base_model_config.yaml` | 9 features sem duplicatas, 3 class weights positivos, seq_len ∈ {720,1440}, optimizer/scheduler válidos, patience ≥ 1 |
| `TestTrainingConfig` | `training_config.yaml` | Chaves obrigatórias, `labelled_dir` aponta para experimento, `model_output` é `.pt`, `batch_size ≤ 512`, `lr ∈ [1e-6, 0.1]`, `dropout ∈ [0, 1)` |
| `TestOptimizationConfig` | `optimization_config.yaml` | `n_trials > 0`, `metric == f1_macro`, `batch_size ≤ 256` (OOM guard), `study_name` definido |
| `TestAuditorConfig` | `auditor_config.yaml` | `n_folds ≥ 3`, XGBoost params positivos/bounded, checkpoint é `.pt`, `xgb_output` é `.json` |
| `TestCrossConfigConsistency` | todos | `labelled_dir` igual em training e auditor; `num_features == len(feature_names)`; `len(class_weights) == num_classes` |

---

### `test_cloud_etl_output.py`

Valida arquivos em `data/L2/pre_processed/` usando **pandas** (parametrizado por arquivo).

| Teste | Descrição |
|:---|:---|
| `test_column_count` | Exatamente 810 colunas (200 bids×2 + 200 asks×2 + 9 features + close) |
| `test_essential_columns` | As 9 features + `close` presentes |
| `test_orderbook_sorting` | Bids decrescentes, Asks crescentes |
| `test_no_book_crossing` | `bid_0_p < ask_0_p` em todas as linhas |
| `test_data_quality_no_nans_in_features` | Zero NaNs nas 9 features |
| `test_chronological_order` | Timestamps monotonicamente crescentes |

---

### `test_preprocessed_quality.py`

Valida `data/L2/pre_processed/` usando **polars** — inclui checks de continuidade em nível de dataset.

| Teste | Descrição |
|:---|:---|
| `test_directory_exists` | Diretório presente |
| `test_file_count` | Pelo menos 1 arquivo |
| `test_date_continuity` | Nenhum dia faltando entre a primeira e última data |
| `test_file_integrity` (param) | ≥ 1.400 linhas/arquivo, schema das 9 features, zero nulls, timestamps monotônicos |

---

### `test_labelling_output.py`

Valida os arquivos Parquet de `labelled_*/` usando **polars**.

Diretório padrão: `ACTIVE_LABELLED_DIR` do conftest. Pode ser sobrescrito via:
```bash
pytest tests/test_labelling_output.py --labelled-dir data/L2/labelled_SELL_0004_BUY_0006_1h
```

| Classe de Testes | Descrição |
|:---|:---|
| `TestLabelledDirectory` | Dir existe; contém Parquets; count bate com `pre_processed/` |
| `TestLabelledFileIntegrity` | `target` presente + int; sem nulls; só {0,1,2}; ≥ 2 classes; 9 features sem NaN; ≥ 1.380 linhas; timestamps monotônicos |
| `TestGlobalLabelBalance` | Sobre 30 dias: todas 3 classes presentes; SELL ≥ 3%, BUY ≥ 3% do total |

---

## 🗺️ Matriz Cobertura × Pipeline

| Componente | Unitário | Config | Dados |
|:---|:---:|:---:|:---:|
| `ETL / pre_processamento` | — | `test_config_integrity` | `test_cloud_etl_output`, `test_preprocessed_quality` |
| `Labelling` | — | `test_config_integrity` | `test_labelling_output` |
| `Hybrid_TCN_LSTM` | `test_model` | `test_config_integrity` | — |
| `Feature Engineering Meta` | `test_meta_features` | — | — |
| `XGBoost Auditor` | — | `test_config_integrity` | — |

> Testes de integração do XGBoost e do `binance_adapter.py` requerem GPU e conectividade — validação manual no RunPod.

---

## ✅ Requisitos

```bash
pip install -r requirements.txt   # inclui pytest, polars, pandas, numpy, pyyaml

# Opcional — unitários do modelo
pip install torch                 # sem GPU: CPU build suficiente para os testes unitários
```

---

## ⚙️ Configuração do pytest (`pytest.ini` / `pyproject.toml`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -ra --tb=short
```
