# 🧪 Guia de Gerenciamento de Experimentos (Auto-Discovery)

Este documento explica como funciona o sistema de **Total Dynamicity** do QuantGod, que permite alterar parâmetros de rotulagem sem a necessidade de atualizar manualmente os caminhos em múltiplos arquivos de configuração.

---

## 🧠 1. Como funciona a Mágica "AUTO"

O sistema utiliza um utilitário central chamado `experiment_utils.py`. Quando um arquivo de configuração (YAML) contém o valor `"AUTO"` nos campos de diretório, o Python resolve o caminho real em tempo de execução seguindo esta hierarquia:

1.  **Leitura do Threshold:** O script lê `src/cloud/base_model/labelling/labelling_config.yaml`.
2.  **Criação do Sufixo:** Com base nos valores de `threshold_long`, `threshold_short` e `lookahead`, ele gera o sufixo (ex: `_SELL_0004_BUY_0004_1h`).
3.  **Localização da Pasta:** Ele busca na pasta `data/L2` pela pasta rotulada que corresponde a esse sufixo.
4.  **Resolução de Splits:** Se o script for de treinamento (Optuna/Especialista), ele automaticamente redireciona para a subpasta dentro de `splits_.../train` ou `splits_.../val`.

---

## 📂 2. Onde buscar e o que alterar

Se você quiser mudar a estratégia de treinamento (ex: ser mais agressivo nos sinais), siga este fluxo:

### Passo A: Definir novos Thresholds
**Arquivo:** `src/cloud/base_model/labelling/labelling_config.yaml`
- Altere `threshold_long` (ex: de `0.004` para `0.005`).
- Altere `lookahead` se desejar prever um horizonte maior.

### Passo B: Rodar o Labelling e o Split
Ao rodar os comandos de processamento, novas pastas serão criadas:
```bash
python src/cloud/base_model/labelling/run_labelling.py
python src/cloud/base_model/treino/create_specialized_splits.py
```
*Note que você não precisa mudar o nome da pasta nos comandos; o código gera o nome correto dinamicamente.*

### Passo C: Otimização e Treino
**Arquivos:** 
- `src/cloud/base_model/otimizacao/optimization_config.yaml`
- `src/cloud/base_model/treino/training_config.yaml`

**O que alterar:** **Absolutamente nada.**
Certifique-se apenas de que eles continuam configurados como:
```yaml
paths:
  train_dir: "AUTO"
  val_dir: "AUTO"
```
O Optuna e o Treino Especialista lerão o novo `labelling_config.yaml` e saberão exatamente onde buscar os novos dados que você acabou de gerar.

---

## 🛠️ 3. Arquivos de Infraestrutura (Evite alterar)

Estes arquivos compõem o motor do sistema. Só altere se quiser mudar a lógica de nomenclatura do projeto:

*   **`src/cloud/base_model/utils/experiment_utils.py`:** Contém a lógica de resolução de caminhos.
*   **`tests/conftest.py`:** Usa o utilitário acima para garantir que os testes rodem sempre contra o dataset ativo.
*   **`src/cloud/base_model/utils/logging_utils.py`:** Define como o sufixo é formatado (`get_labelling_suffix`).

---

## ✅ 4. Checklist de Verificação
Para ter certeza de que o sistema está "sincronizado", você pode rodar:
```bash
pytest tests/labelling/test_labelling_output.py -v
```
Se o teste passar, significa que a "mão invisível" do `experiment_utils` encontrou sua pasta nova e validou os dados corretamente.
