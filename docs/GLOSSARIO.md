# 📖 Glossário de ativos — QuantGod Cloud ⚡

Este documento serve como um mapa central para localizar e entender a finalidade de cada métrica e arquivo no ecossistema QuantGod v5.0.

---

## 🏗️ 1. Terminologia do Domínio da Informação

| Termo | Definição |
|:---|:---|
| **Dollar Bar** | Amostragem que fecha uma barra quando um volume fixo em USD é transacionado. |
| **Tick Bar** | Amostragem baseada em um número fixo de negócios (trades), independente do volume. |
| **Information Bar** | Barra que fecha baseada no desequilíbrio do fluxo (OFI/VPIN). Segue a lógica de amostragem de eventos raros. |
| **IID Bars** | Barras Independents and Identically Distributed. Garantem que a variância estatística dos dados seja constante. |
| **Reconciliação de Transbordo** | Técnica para levar o excedente de volume/ticks de uma barra para a próxima, garantindo conservação de massa. |
| **Scale Guard** | Lógica de resiliência que adapta o número de workers baseando-se na densidade de dados do ano (ex: 2026) para evitar OOM. |

---

## 🤖 2. Modelos e Pesos (`data/models/`)

| Arquivo/Pasta | Descrição |
|:---|:---|
| `base_model.pt` | **Modelo Fundação (TCN+LSTM)**: Treinado em barras IID para capturar dependências temporais longas. |
| **Specialist Model** | Modelo refinado via K-Fold e Big Data focado em maximizar o Sniper Score. |
| **Lazy Loading** | Carregamento sob demanda de Parquets via `QuantGodLazyDataset` para suportar 345M amostras. |
| `xgb_auditor.json` | **Auditor XGBoost**: Modelo que valida a confiança das predições do modelo base. |
| `scaler_finetuning.pkl` | Normalizador central (StandardScaler) das 32 features snipers. |

---

## 🏷️ 3. Estratégia de Rotulagem (Prado-IID)

| Termo | Definição |
|:---|:---|
| **Triple Barrier** | Método que usa barreiras de Take Profit, Stop Loss e Tempo para rotular o futuro do preço. |
| **EWMA Volatility** | Volatilidade adaptativa usada para definir a largura das barreiras de TP/SL. |
| **Sniper Score** | Métrica híbrida (70% direcional + 30% macro) usada para otimizar o Especialista. |
| **Ambiguity Filter** | Filtro que marca como Neutro (1) amostras onde o preço atingiu tanto o TP quanto o SL na mesma janela. |

---

## 📊 4. Estrutura de Dados (`data/L2/`)

| Pasta | Formato | Conteúdo |
|:---|:---|:---|
| `raw/` | `.zip` / `.csv.gz` | Dados brutos de L2 e Trades diretos da exchange. |
| `pre_processed/` | `.parquet` | Dados após ETL v5.0 (833 colunas, amostragem de eventos). |
| `labelled_*/` | `.parquet` | Datasets prontos para treino com labels Triple Barrier. |

---

## ⚙️ 5. Configurações (`src/cloud/base_model/configs/`)

| Arquivo | Uso |
|:---|:---|
| `master_config.yaml` | **Fonte Única de Verdade**. Controla thresholds de barras, multiplicadores de barreiras e paths. |

---

> **Dica**: Em caso de dúvida sobre uma feature específica, consulte o documento [`7_DATA_REFERENCE.md`](7_DATA_REFERENCE.md).
