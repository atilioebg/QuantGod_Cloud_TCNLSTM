# Arquitetura Out-Of-Fold (OOF) do Modelo Especialista (Sniper v4.5+)

Este documento detalha o fluxo de execução técnica quando a flag `kfold.enabled: true` está ativada no `master_config.yaml`.

## O Problema do Vazamento de Dados no Fluxo Legado
No fluxo clássico de Otimização (Foundation -> Especialista -> Auditor XGBoost), treinar o Especialista (Fase 2) em um split contínuo tradicional de 80%/20% apresenta uma falha conceitual grave para a Fase 3:
Todo o dataset de predições que o Juiz Auditor lerá terá "enxergado" dados de treinamento. As correlações das features já estarão absorvidas. Quando o XGBoost tenta aprender "quando o TCN fracassa", ele não enxerga a verdadeira fragilidade e sim um *Overfitting* mitigado artificialmente.

## A Solução: Blocked Purged K-Fold
Para alimentar o modelo final do Auditor com Sinais **100% Cíbaros (Blindados)**, usamos o algoritmo de Blocked Purged K-Fold habilitado por padrão.

### Como funciona passo a passo:

#### 1. Herança Arquitetural (DNA Base)
O pipeline **NÃO RODA O OPTUNA NOVAMENTE** no Especialista K-Fold para poupar centenas de horas de GPU. Ao invés disso, o script `run_kfold_specialist.py` carrega intimamente o arquivo físico `best_params.json` (A topologia vencedora da Fase 1, já consolidando Canis TCN, LSTMs, Batch Size, etc) como se fosse o esqueleto base de um carro de corrida provado.

#### 2. Focal Loss "Hiper-Punitiva" (Alma Sniper)
O que muda no K-Fold são as "lentes". O modelo assume os pesos drásticos definidos pela matriz `specialization_weights`:
- **`spec_alpha_neutral`** (Ex: 0.1 a 0.8) esmaga o peso do alvo majoritário (Neutro). O modelo pára de ligar pra zonas mortas.
- **`spec_alpha_side`** (Ex: 1.5 a 6.0) amplifica em milhares de vezes o erro ao não prever uma Compra/Venda válida.
- O limiar de Smoothing despenca visando "cravadas direcionais agressivas".

#### 3. Segmentação Temporal com Guilhotina (Purge Gap)
O gigantesco conjunto L2 é rasgado em `N` fatias contínuas de tempo (Ex: Gêmeos de 5 Blocos T1, T2, T3, T4, T5).
No Fold 1, a rodada de Teste é o array `T1`, logo a rodada de treinamento seria `T2+T3+T4+T5`. Contudo, as junções dessas placas temporais cruzam o mesmo horizonte de eventos que contêm features retrospectivas gigantescas (ex: *vwap_zscore* de 2h ou *delta_vol_24h*).
- O Script KFold aplica um recuo matemático exato de N Linhas (igual a `kfold.purge_minutes` do YAML).
- O Final de `T1` ou Frente do `T2` se tornam Buracos Negros (Void), **impossibilitando o vazamento por contexto ou contiguidade**.

#### 4. O Treino Múltiplo Isolado
O script repetirá o ciclo de Treinamento **N (5)** vezes:
1. Nasce "Bebê Módulo M1". Treina (T2, T3, T4, T5). Inspeciona (T1). Salva Logits Inspecionados (`fold_0.parquet`). MORRE. LIMPA VRAM cuda.
2. Nasce "Bebê Módulo M2". Treina (T1, T3, T4, T5). Inspeciona (T2). Salva Logits Inspecionados (`fold_1.parquet`). MORRE. LIMPA VRAM cuda.
3. Até completar os `N_Splits`.

#### 5. A Combinação Genética (Full OOF)
Ao final, a função `concatenate_oofs_to_single_file` junta os 5 pratos do banquete (fold_0 a fold_4).
O resultado é um arquivo sagrado em `data/auditor/oof_predictions/full_oof.parquet`. Ele engloba a Totalidade dos Dias mapeados do CSV, mas **GARANTINDO** que a probabilidade `[BUY, NEUTRAL, SELL]` alocada em cada linha específica foi gerada por uma Máquina que **NUNCA**, sob qualquer viés, havia treinado sobre aquela hora de mercado.

É nesse arquivo imaculado que o **Juiz XGBoost** (Fase 3) será treinado, providenciando proteção de patrimônio simulada perfeitamente no *Backtrader*.
