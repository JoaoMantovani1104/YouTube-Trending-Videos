# Guia de Execução Simplificado

Este guia explica como usar o script `divisao_modelagem_avaliacao.py` para treinar um modelo por vez e salvar os resultados de forma rastreável em uma pasta personalizada dentro de `resultados_avaliacao/`.

## 1. Sintaxe geral

A sintaxe exige que você informe o modelo (`--model`), parâmetros específicos do modelo (`--params`) e configurações globais (K-Fold, tamanho do teste, etc.).

Exemplo de uso (Bash):

```bash
python3 divisao_modelagem_avaliacao.py \
    --model [NOME_MODELO] \
    --params [CHAVE1=VALOR1,CHAVE2=VALOR2] \
    --k_folds [K_FOLDS] \
    --test_size [TAMANHO_TESTE]
```

## 2. Exemplos de execução por modelo

### 2.1 Random Forest Regressor

Para executar o Random Forest Regressor com 100 estimadores, profundidade máxima 10, K-Fold=5 e tamanho do teste 20%:

```bash
python3 divisao_modelagem_avaliacao.py \
    --model RandomForest \
    --params n_estimators=100,max_depth=10,min_samples_split=5 \
    --k_folds 5 \
    --test_size 0.2
```

Exemplo com mais estimadores e sem limite de profundidade:

```bash
python3 divisao_modelagem_avaliacao.py \
    --model RandomForest \
    --params n_estimators=200,min_samples_split=2 \
    --k_folds 10 \
    --test_size 0.15
```

### 2.2 Gradient Boosting Regressor

Para executar o Gradient Boosting com taxa de aprendizado 0.05, profundidade máxima 5 e 100 estimadores:

```bash
python3 divisao_modelagem_avaliacao.py \
    --model GradientBoosting \
    --params learning_rate=0.05,max_depth=5,n_estimators=100 \
    --k_folds 5 \
    --test_size 0.2
```

Exemplo com configuração mais conservadora (menor taxa de aprendizado):

```bash
python3 divisao_modelagem_avaliacao.py \
    --model GradientBoosting \
    --params learning_rate=0.01,max_depth=3,n_estimators=200 \
    --k_folds 10 \
    --test_size 0.2
```

### 2.3 Ridge Regression

Para executar a Regressão Ridge com regularização alpha=1.0:

```bash
python3 divisao_modelagem_avaliacao.py \
    --model Ridge \
    --params alpha=1.0 \
    --k_folds 5 \
    --test_size 0.2
```

Exemplo com regularização mais forte:

```bash
python3 divisao_modelagem_avaliacao.py \
    --model Ridge \
    --params alpha=10.0 \
    --k_folds 10 \
    --test_size 0.15
```

## 3. Parâmetros por modelo

### Random Forest
- `n_estimators`: Número de árvores na floresta (ex: 100, 200, 500)
- `max_depth`: Profundidade máxima das árvores (ex: 5, 10, 15, ou omitir para sem limite)
- `min_samples_split`: Número mínimo de amostras para dividir um nó (ex: 2, 5, 10)

### Gradient Boosting
- `learning_rate`: Taxa de aprendizado (ex: 0.01, 0.05, 0.1, 0.2)
- `n_estimators`: Número de estimadores de boosting (ex: 100, 200, 300)
- `max_depth`: Profundidade máxima das árvores individuais (ex: 3, 5, 7)

### Ridge
- `alpha`: Força da regularização L2 (ex: 0.1, 1.0, 10.0, 100.0)

## 4. Resultado da execução

Após a execução, o script criará uma pasta rastreável dentro de `resultados_avaliacao/`. O nome da pasta incluirá o nome do modelo, os hiperparâmetros e as configurações usadas (por exemplo: seed, amostra usada, tamanho do teste).

Exemplo de estrutura gerada:

```
resultados_avaliacao/RandomForest_ne100_md10_mss5_5k_02test_42rs/
├── 04_documentacao_metodologia.txt    # descrição do que foi feito
├── 05_metricas_avaliacao.csv         # métricas de desempenho (treino/val/test)
└── 06_importance.png                 # importância de features (quando aplicável)
```

## 5. Dicas rápidas

**Observações sobre `--params`:**
- Separe pares `chave=valor` por vírgula, sem espaços.
- Os nomes das chaves dependem do modelo (veja seção 3).
- Valores decimais usam ponto como separador (ex: `learning_rate=0.05`).

**Configurações gerais:**
- Use `--k_folds` para validação cruzada; escolha 5 ou 10, conforme tamanho do conjunto.
- Defina `--test_size` como porcentagem (ex.: `0.2` para 20%).
- Inclua um `random_state` nos `--params` para reprodutibilidade quando suportado.
- Consulte os nomes de parâmetro aceitos no código-fonte do projeto para garantir compatibilidade.

## Contato
Para dúvidas ou problemas, abra uma issue no repositório com o comando usado e o log de execução.

---
Arquivo gerado automaticamente com instruções de execução.