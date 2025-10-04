import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# 0. Configuração e Parsing de Argumentos da Linha de Comando
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Executa o pipeline de modelagem e avaliação com parâmetros passados via CLI.")

    # Argumentos Globais (Método)
    parser.add_argument('--model', type=str, required=True,
                        choices=['RandomForest', 'GradientBoosting', 'Ridge'],
                        help="Modelo a ser treinado (RandomForest, GradientBoosting, ou Ridge).")
    parser.add_argument('--k_folds', type=int, default=5,
                        help="Número de folds para K-Fold Cross-Validation (K=5 ou 10).")
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="Porcentagem dos dados reservados para teste (ex: 0.2 para 20%%).")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Semente aleatória para reprodutibilidade.")

    # Argumentos Específicos do Modelo (Hiperparâmetros)
    parser.add_argument('--params', type=str, required=True,
                        help="Hiperparâmetros no formato chave1=valor1,chave2=valor2...")

    return parser.parse_args()

# --- Funções Auxiliares ---

def parse_params_string(params_string):
    """Converte a string de parâmetros (ex: 'a=0.1,b=1') em um dicionário de Grid Search."""
    params = {}
    for item in params_string.split(','):
        if '=' in item:
            key, value = item.split('=')
            try:
                # Tenta converter para int, float ou mantém como string
                if '.' in value:
                    params[key.strip()] = float(value.strip())
                else:
                    params[key.strip()] = int(value.strip())
            except ValueError:
                params[key.strip()] = value.strip()
    # Os grids de Grid Search devem ser listas
    return {k: [v] for k, v in params.items()}

def create_param_suffix(model_name, params, k_folds, test_size, random_state):
    """Cria a string compacta com todos os parâmetros para o nome da pasta/arquivo."""
    suffixes = {'max_depth': 'md', 'n_estimators': 'ne', 'min_samples_split': 'mss', 'learning_rate': 'lr', 'alpha': 'a'}
    param_list = []

    # Usa o valor dentro da lista do Grid Search (que só tem 1 elemento)
    for key, value_list in params.items():
        value = value_list[0]
        suf = suffixes.get(key, key[0])
        formatted_value = str(value).replace('.', '')
        param_list.append(f"{suf}{formatted_value}")

    param_str = "_".join(param_list)

    global_suffix = f"_{k_folds}k_{str(test_size).replace('.', '')}test_{random_state}rs"

    return f"{model_name}_{param_str}{global_suffix}"

def get_model(model_name, params, random_state):
    """Retorna o modelo inicializado, pronto para o Grid Search."""
    if model_name == 'RandomForest':
        return RandomForestRegressor(random_state=random_state)
    elif model_name == 'GradientBoosting':
        return GradientBoostingRegressor(random_state=random_state)
    elif model_name == 'Ridge':
        return Ridge(random_state=random_state)
    else:
        raise ValueError(f"Modelo {model_name} não suportado.")

def evaluate_model(model, X_set, Y_true, set_name):
    """Calcula as métricas obrigatórias de regressão (RMSE, MAE, R2)."""
    Y_pred = model.predict(X_set)
    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    return {'Set': set_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ==============================================================================
# 1. Fluxo Principal
# ==============================================================================

if __name__ == '__main__':
    args = parse_args()

    # 1.1 Variáveis de Configuração
    MODEL_NAME = args.model
    K_FOLDS = args.k_folds
    TEST_SIZE = args.test_size
    RANDOM_STATE = args.random_state

    # O Grid Search aqui só testará 1 combinação: a passada na linha de comando
    PARAM_GRID = parse_params_string(args.params)
    OUTPUT_DIR = 'resultados_avaliacao'

    # 1.2 Carregamento e Divisão dos Dados (Seção 4)
    output_filename = 'pre_processed_youtube_trending_data.csv'
    try:
        df_processed = pd.read_csv(output_filename)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{output_filename}' não encontrado. Execute o pré-processamento primeiro.")
        exit()

    Y = df_processed['views']
    cols_to_drop_final = ['video_id', 'title', 'channel_title', 'publish_time', 'tags', 'description', 'thumbnail_link', 'views']
    X = df_processed.drop(columns=[col for col in cols_to_drop_final if col in df_processed.columns], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 1.3 Criação da Pasta de Saída Personalizada
    custom_suffix = create_param_suffix(MODEL_NAME, PARAM_GRID, K_FOLDS, TEST_SIZE, RANDOM_STATE)
    FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, custom_suffix)

    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)
        print(f"\nSubpasta de resultados criada: {FINAL_OUTPUT_DIR}")
    else:
        print(f"\nSubpasta já existe. Resultados serão sobrescritos em: {FINAL_OUTPUT_DIR}")

    # 1.4 Modelagem e Otimização (Seção 5)
    model = get_model(MODEL_NAME, PARAM_GRID, RANDOM_STATE)
    scoring_metric = 'neg_mean_squared_error'

    # Usando Grid Search para treinar o modelo na ÚNICA combinação passada
    grid_search = GridSearchCV(estimator=model, param_grid=PARAM_GRID, cv=kf, scoring=scoring_metric, verbose=1, n_jobs=-1)
    print(f"Iniciando treinamento para {MODEL_NAME} com parâmetros: {args.params}")
    grid_search.fit(X_train, Y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 1.5 Avaliação (Seção 6)
    evaluation_results = []
    evaluation_results.append(evaluate_model(best_model, X_train, Y_train, 'Treino'))
    evaluation_results.append(evaluate_model(best_model, X_test, Y_test, 'Teste'))

    df_results = pd.DataFrame(evaluation_results)
    df_results_pivot = df_results.pivot_table(index='Set', values=['RMSE', 'MAE', 'R2'])

    # ==============================================================================
    # 1.6 Apresentação e Salvamento dos Resultados (REFORMULADO)
    # ==============================================================================

    print("\n" + "="*70)
    print(f"RESUMO DA EXECUÇÃO - Modelo: {MODEL_NAME}")
    print("="*70)

    # Impressão da Configuração
    print("\n--- Configuração de Treinamento ---")
    print(f"Divisão: Treino={1 - TEST_SIZE:.0%} / Teste={TEST_SIZE:.0%} | K-Fold (CV)={K_FOLDS}")
    print(f"Semente Aleatória (Random State): {RANDOM_STATE}")

    # Impressão dos Hiperparâmetros
    print("\n--- Melhores Hiperparâmetros Encontrados (Seção 5) ---")
    print(json.dumps(best_params, indent=4))

    # Impressão das Métricas de Regressão (RMSE, MAE, R2)
    print("\n--- Métricas de Avaliação (Seção 6) ---")
    # Formata os valores para melhor visualização (3 casas decimais)
    df_results_formatted = df_results_pivot.apply(lambda x: x.apply('{:,.3f}'.format))
    print(df_results_formatted)

    # Análise Rápida de Overfitting
    r2_train = df_results_pivot.loc['Treino', 'R2']
    r2_test = df_results_pivot.loc['Teste', 'R2']
    print("\n--- Análise de Overfitting (R2) ---")
    print(f"R2 (Treino): {r2_train:.4f} | R2 (Teste): {r2_test:.4f}")

    if (r2_train - r2_test) > 0.05 and r2_train > 0.8:
        print("⚠️ Atenção: Potencial de Overfitting! Grande diferença entre R2 de Treino e Teste.")
    elif r2_test < 0.5:
        print("🔍 O modelo apresenta baixo poder de predição (R2 baixo) no conjunto de teste.")
    else:
        print("✅ Desempenho balanceado entre Treino e Teste.")

    # Salvando Hiperparâmetros (Documentação)
    output_params_path = os.path.join(FINAL_OUTPUT_DIR, f'04_documentacao_metodologia.txt')
    with open(output_params_path, 'w') as f:
        f.write("--- CONFIGURAÇÃO DE EXECUÇÃO ---\n")
        f.write(f"Modelo: {MODEL_NAME}\n")
        f.write(f"Divisão: Treino={1 - TEST_SIZE:.0%} / Teste={TEST_SIZE:.0%} | K-Fold={K_FOLDS} | Random State={RANDOM_STATE}\n")
        f.write("\nMelhores Parâmetros (Execução Atual):\n")
        f.write(json.dumps(best_params, indent=4))
        f.write("\n")

    # Salvando Métricas
    output_table_path = os.path.join(FINAL_OUTPUT_DIR, '05_metricas_avaliacao.csv')
    df_results_pivot.to_csv(output_table_path)
    print(f"\n✅ Métricas de Avaliação salvas em: {output_table_path}")

    # 1.7 Explicabilidade (Seção 7)
    if MODEL_NAME in ['RandomForest', 'GradientBoosting']:
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        # Seleciona as 10 principais para o gráfico de barras
        feature_importance[:10].plot(kind='barh')
        plt.title(f'Top 10 Importância de Features - Modelo: {MODEL_NAME}')
        plt.tight_layout()
        output_plot_path = os.path.join(FINAL_OUTPUT_DIR, '06_importance.png')
        plt.savefig(output_plot_path)
        plt.close()
        print(f"✅ Gráfico de Importância de Features salvo em: {output_plot_path}")
    elif MODEL_NAME == 'Ridge':
         print("\nModelo Ridge selecionado: Use coeficientes (coef_) para análise de importância (explicabilidade).")

    print("\n" + "="*70)
    print("EXECUÇÃO CONCLUÍDA. Verifique a pasta de resultados para os arquivos.")
    print("="*70)