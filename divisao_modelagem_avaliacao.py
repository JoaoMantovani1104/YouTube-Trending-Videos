import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===========================================
# FUNÇÕES AUXILIARES
# ===========================================

def print_titulo(titulo):
    """Imprime um título bonito"""
    print("\n" + "="*60)
    print(f"🔹 {titulo}")
    print("="*60)

def print_secao(numero, titulo):
    """Imprime uma seção numerada"""
    print(f"\n📍 {numero}. {titulo}")
    print("-" * 40)

def solicitar_modelo():
    """Solicita interativamente qual modelo usar"""
    print_titulo("CONFIGURAÇÃO INTERATIVA DO MODELO")
    print("\n🤖 Modelos disponíveis:")
    print("   1️⃣  Ridge (Regressão Linear Regularizada)")
    print("   2️⃣  RandomForest (Árvores Aleatórias)")
    print("   3️⃣  GradientBoosting (Gradient Boosting Regressor)")

    while True:
        escolha = input("\n👉 Escolha o modelo (1, 2 ou 3): ").strip()
        if escolha == '1':
            return 'Ridge'
        elif escolha == '2':
            return 'RandomForest'
        elif escolha == '3':
            return 'GradientBoosting'
        else:
            print("❌ Opção inválida! Digite 1, 2 ou 3.")

def solicitar_parametros_ridge():
    """Solicita parâmetros para o modelo Ridge"""
    print("\n🔧 Configurando Ridge:")
    print("   📋 Alpha: controla a regularização (quanto maior, mais simples o modelo)")
    print("   💡 Sugestões: 0.1 (pouca), 1.0 (média), 10.0 (alta regularização)")

    while True:
        try:
            alpha = float(input("\n👉 Digite o valor de alpha (ex: 10.0): "))
            if alpha > 0:
                return {'alpha': alpha}
            else:
                print("❌ Alpha deve ser maior que zero!")
        except ValueError:
            print("❌ Digite um número válido!")

def solicitar_parametros_randomforest():
    """Solicita parâmetros para o modelo RandomForest"""
    print("\n🔧 Configurando RandomForest:")

    # N_estimators
    print("\n🌳 N_estimators: número de árvores na floresta")
    print("   💡 Sugestões: 50 (rápido), 100 (padrão), 200 (mais preciso)")
    while True:
        try:
            n_est = int(input("👉 Digite n_estimators (ex: 100): "))
            if n_est > 0:
                break
            else:
                print("❌ Deve ser maior que zero!")
        except ValueError:
            print("❌ Digite um número inteiro!")

    # Max_depth
    print("\n🌲 Max_depth: profundidade máxima das árvores")
    print("   💡 Sugestões: 5 (simples), 10 (padrão), None (sem limite)")
    while True:
        depth_input = input("👉 Digite max_depth (ex: 10) ou 'None': ").strip()
        if depth_input.lower() == 'none':
            max_depth = None
            break
        else:
            try:
                max_depth = int(depth_input)
                if max_depth > 0:
                    break
                else:
                    print("❌ Deve ser maior que zero!")
            except ValueError:
                print("❌ Digite um número inteiro ou 'None'!")

    return {'n_estimators': n_est, 'max_depth': max_depth, 'random_state': 42}

def solicitar_parametros_gradientboosting():
    """Solicita parâmetros para o modelo GradientBoosting"""
    print("\n🔧 Configurando GradientBoosting:")

    # N_estimators
    print("\n🌳 N_estimators: número de árvores de boosting")
    print("   💡 Sugestões: 50 (rápido), 100 (padrão), 200 (mais preciso)")
    while True:
        try:
            n_est = int(input("👉 Digite n_estimators (ex: 100): "))
            if n_est > 0:
                break
            else:
                print("❌ Deve ser maior que zero!")
        except ValueError:
            print("❌ Digite um número inteiro!")

    # Learning rate
    print("\n📈 Learning_rate: taxa de aprendizado (shrinkage)")
    print("   💡 Sugestões: 0.01 (lento/preciso), 0.1 (padrão), 0.2 (rápido)")
    while True:
        try:
            lr = float(input("👉 Digite learning_rate (ex: 0.1): "))
            if 0 < lr <= 1:
                break
            else:
                print("❌ Deve ser entre 0 e 1!")
        except ValueError:
            print("❌ Digite um número válido!")

    # Max_depth
    print("\n🌲 Max_depth: profundidade máxima das árvores")
    print("   💡 Sugestões: 3 (simples), 6 (padrão), 10 (complexo)")
    while True:
        try:
            max_depth = int(input("👉 Digite max_depth (ex: 6): "))
            if max_depth > 0:
                break
            else:
                print("❌ Deve ser maior que zero!")
        except ValueError:
            print("❌ Digite um número inteiro!")

    return {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': max_depth, 'random_state': 42}

def solicitar_configuracoes_gerais():
    """Solicita configurações gerais do treinamento"""
    print("\n⚙️  Configurações Gerais:")

    # Tamanho do teste
    print("\n📊 Tamanho do conjunto de teste:")
    print("   💡 Sugestões: 0.2 (20%), 0.3 (30%)")
    while True:
        try:
            test_size = float(input("👉 Digite a porcentagem para teste (ex: 0.2): "))
            if 0 < test_size < 1:
                break
            else:
                print("❌ Deve ser entre 0 e 1!")
        except ValueError:
            print("❌ Digite um número válido!")

    # Random state
    print("\n🎲 Random State (semente aleatória para reprodutibilidade):")
    print("   💡 Sugestões: 42 (padrão), 123, qualquer número")
    while True:
        try:
            random_state = int(input("👉 Digite o random_state (ex: 42): "))
            break
        except ValueError:
            print("❌ Digite um número inteiro!")

    return test_size, random_state

def criar_nome_execucao(modelo, parametros, tamanho_teste, random_state):
    """Cria um nome único para a execução baseado nos parâmetros"""
    # Mapeamento de abreviações para parâmetros
    abreviacoes = {
        'alpha': 'a',
        'n_estimators': 'ne',
        'max_depth': 'md',
        'learning_rate': 'lr'
    }

    # Construir string dos parâmetros
    param_parts = []
    for chave, valor in parametros.items():
        if chave != 'random_state':  # Não incluir random_state nos parâmetros
            abrev = abreviacoes.get(chave, chave[:2])
            if valor is None:
                param_parts.append(f"{abrev}None")
            else:
                # Remover pontos dos números decimais
                valor_str = str(valor).replace('.', '')
                param_parts.append(f"{abrev}{valor_str}")

    param_str = "_".join(param_parts)
    test_str = str(tamanho_teste).replace('.', '')

    # Timestamp para garantir unicidade
    timestamp = int(time.time()) % 10000  # Últimos 4 dígitos do timestamp

    return f"execucao_{timestamp}_{param_str}_{test_str}test_{random_state}rs"

def contar_execucoes_modelo(modelo):
    """Conta quantas execuções já existem para um modelo"""
    pasta_modelo = modelo
    if os.path.exists(pasta_modelo):
        return len([d for d in os.listdir(pasta_modelo) if os.path.isdir(os.path.join(pasta_modelo, d))])
    return 0

def imprimir_info_dataset(df):
    """Imprime informações detalhadas sobre o dataset"""
    print_secao("INFO", "INFORMAÇÕES DO DATASET")

    print(f"📏 Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"💾 Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Tipos de dados
    print(f"\n📋 Tipos de dados:")
    tipos = df.dtypes.value_counts()
    for tipo, count in tipos.items():
        print(f"   {tipo}: {count} colunas")

    # Valores ausentes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n❌ Valores ausentes encontrados:")
        for col, count in missing[missing > 0].items():
            print(f"   {col}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n✅ Sem valores ausentes!")

    # Estatísticas da variável target
    if 'views' in df.columns:
        views = df['views']
        print(f"\n🎯 Estatísticas da variável target (views):")
        print(f"   📊 Média: {views.mean():,.0f}")
        print(f"   📊 Mediana: {views.median():,.0f}")
        print(f"   📊 Mínimo: {views.min():,.0f}")
        print(f"   📊 Máximo: {views.max():,.0f}")
        print(f"   📊 Desvio Padrão: {views.std():,.0f}")

def calcular_e_exibir_metricas(y_real, y_pred, nome_conjunto, cor_emoji):
    """Calcula e exibe métricas de avaliação"""
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)

    print(f"\n{cor_emoji} {nome_conjunto.upper()}:")
    print(f"   📏 RMSE (Erro Quadrático Médio): {rmse:,.0f}")
    print(f"   📐 MAE (Erro Absoluto Médio):    {mae:,.0f}")
    print(f"   📊 R² (Coeficiente Determinação): {r2:.4f} ({r2*100:.1f}%)")

    # Interpretação do R²
    if r2 >= 0.9:
        interpretacao = "🌟 Excelente!"
    elif r2 >= 0.7:
        interpretacao = "👍 Muito bom!"
    elif r2 >= 0.5:
        interpretacao = "🔧 Razoável"
    else:
        interpretacao = "❌ Fraco"

    print(f"   💡 Interpretação: {interpretacao}")

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ===========================================
# CONFIGURAÇÃO INTERATIVA
# ===========================================

print_titulo("🚀 SISTEMA DE MODELAGEM INTERATIVO")
print("Bem-vindo ao sistema de Machine Learning para predição de views do YouTube!")

# Solicitar configurações interativamente
MODELO = solicitar_modelo()

if MODELO == 'Ridge':
    PARAMETROS = solicitar_parametros_ridge()
elif MODELO == 'RandomForest':
    PARAMETROS = solicitar_parametros_randomforest()
elif MODELO == 'GradientBoosting':
    PARAMETROS = solicitar_parametros_gradientboosting()

TAMANHO_TESTE, RANDOM_STATE = solicitar_configuracoes_gerais()

# ===========================================
# CARREGAMENTO DOS DADOS
# ===========================================

print_secao("1", "CARREGAMENTO DOS DADOS")
try:
    df = pd.read_csv('pre_processed_youtube_trending_data.csv')
    print(f"✅ Arquivo carregado com sucesso!")
    imprimir_info_dataset(df)
except FileNotFoundError:
    print("❌ ERRO: Arquivo 'pre_processed_youtube_trending_data.csv' não encontrado!")
    print("💡 Certifique-se de que o arquivo está no diretório atual.")
    exit()

# ===========================================
# PREPARAÇÃO DOS DADOS
# ===========================================

print_secao("2", "PREPARAÇÃO DOS DADOS")

# Variável alvo (o que queremos prever)
y = df['views']

# Variáveis de entrada (features) - removendo colunas não numéricas
colunas_remover = ['video_id', 'title', 'channel_title', 'publish_time',
                   'tags', 'description', 'thumbnail_link', 'views']
X = df.drop(columns=[col for col in colunas_remover if col in df.columns])

print(f"🎯 Variável target: 'views' ({len(y):,} valores)")
print(f"🔢 Features selecionadas: {X.shape[1]} colunas numéricas")
print(f"\n📋 Lista das features:")
for i, col in enumerate(X.columns, 1):
    print(f"   {i:2d}. {col}")

# Verificar se há valores infinitos ou muito grandes
if np.isinf(X.values).any():
    print("⚠️  Valores infinitos detectados - serão removidos")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

print(f"\n✅ Dados preparados: {X.shape[0]:,} amostras × {X.shape[1]} features")

# ===========================================
# DIVISÃO TREINO/TESTE
# ===========================================

print_secao("3", "DIVISÃO TREINO/TESTE")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=TAMANHO_TESTE, random_state=RANDOM_STATE
)

print(f"📊 Divisão configurada:")
print(f"   🔵 Treino: {len(X_treino):,} amostras ({(1-TAMANHO_TESTE)*100:.0f}%)")
print(f"   🔴 Teste:  {len(X_teste):,} amostras ({TAMANHO_TESTE*100:.0f}%)")
print(f"   🎲 Random State: {RANDOM_STATE}")

# Estatísticas dos conjuntos
print(f"\n📈 Estatísticas da variável target por conjunto:")
print(f"   Treino - Média: {y_treino.mean():,.0f} | Desvio: {y_treino.std():,.0f}")
print(f"   Teste  - Média: {y_teste.mean():,.0f} | Desvio: {y_teste.std():,.0f}")

# ===========================================
# TREINAMENTO DO MODELO
# ===========================================

print_secao("4", f"TREINAMENTO DO MODELO {MODELO}")

print(f"🤖 Modelo selecionado: {MODELO}")
print(f"⚙️  Parâmetros configurados:")
for param, valor in PARAMETROS.items():
    print(f"   {param}: {valor}")

# Criando o modelo
if MODELO == 'Ridge':
    modelo = Ridge(**PARAMETROS)
elif MODELO == 'RandomForest':
    modelo = RandomForestRegressor(**PARAMETROS)
elif MODELO == 'GradientBoosting':
    modelo = GradientBoostingRegressor(**PARAMETROS)

print(f"\n🚀 Iniciando treinamento...")
modelo.fit(X_treino, y_treino)
print(f"✅ Modelo {MODELO} treinado com sucesso!")

# ===========================================
# AVALIAÇÃO
# ===========================================

print_secao("5", "AVALIAÇÃO DO MODELO")

# Fazendo previsões
print("🔮 Fazendo previsões...")
y_pred_treino = modelo.predict(X_treino)
y_pred_teste = modelo.predict(X_teste)

# Calculando métricas
metricas_treino = calcular_e_exibir_metricas(y_treino, y_pred_treino, "treino", "🔵")
metricas_teste = calcular_e_exibir_metricas(y_teste, y_pred_teste, "teste", "🔴")

# ===========================================
# ANÁLISE DE OVERFITTING
# ===========================================

print_secao("6", "ANÁLISE DE OVERFITTING")

diferenca_r2 = metricas_treino['R2'] - metricas_teste['R2']
diferenca_rmse = abs(metricas_treino['RMSE'] - metricas_teste['RMSE'])

print(f"📈 Comparação Treino vs Teste:")
print(f"   Diferença R²:   {diferenca_r2:+.4f}")
print(f"   Diferença RMSE: {diferenca_rmse:,.0f}")

print(f"\n🔍 Diagnóstico:")
if diferenca_r2 > 0.1 and metricas_treino['R2'] > 0.8:
    print("   ⚠️  OVERFITTING DETECTADO!")
    print("   💡 O modelo está decorando os dados de treino")
    print("   🔧 Sugestões: aumentar regularização ou reduzir complexidade")
elif diferenca_r2 > 0.05:
    print("   🟡 Leve overfitting detectado")
    print("   💡 Ainda aceitável, mas monitore")
elif diferenca_r2 < -0.05:
    print("   🟠 Possível underfitting")
    print("   💡 Modelo pode estar muito simples")
else:
    print("   ✅ Modelo bem balanceado!")
    print("   💡 Boa generalização entre treino e teste")

if metricas_teste['R2'] < 0.3:
    print("   📉 Performance geral baixa no teste")
    print("   🔧 Considere: mais features, dados ou modelo diferente")

# ===========================================
# SALVANDO RESULTADOS
# ===========================================

print_secao("7", "SALVANDO RESULTADOS")

# Criando estrutura de pastas organizada
pasta_modelo = MODELO  # Pasta principal do modelo (ex: Ridge, RandomForest, GradientBoosting)
nome_execucao = criar_nome_execucao(MODELO, PARAMETROS, TAMANHO_TESTE, RANDOM_STATE)
pasta_execucao = os.path.join(pasta_modelo, nome_execucao)

# Criar pasta do modelo se não existir
if not os.path.exists(pasta_modelo):
    os.makedirs(pasta_modelo)
    print(f"📁 Pasta do modelo criada: {pasta_modelo}/")

# Criar pasta da execução
os.makedirs(pasta_execucao, exist_ok=True)
num_execucoes = contar_execucoes_modelo(pasta_modelo)
print(f"📂 Execução #{num_execucoes}: {pasta_execucao}/")
print(f"💡 Esta é a {num_execucoes}ª execução do modelo {MODELO}")

# Salvando métricas em CSV
resultados_df = pd.DataFrame({
    'Conjunto': ['Treino', 'Teste'],
    'RMSE': [metricas_treino['RMSE'], metricas_teste['RMSE']],
    'MAE': [metricas_treino['MAE'], metricas_teste['MAE']],
    'R2': [metricas_treino['R2'], metricas_teste['R2']]
})

arquivo_metricas = os.path.join(pasta_execucao, 'metricas_detalhadas.csv')
resultados_df.to_csv(arquivo_metricas, index=False, float_format='%.6f')
print(f"💾 Métricas salvas: {arquivo_metricas}")

# Salvando configurações usadas
config_dict = {
    'execucao': nome_execucao,
    'modelo': MODELO,
    'parametros': PARAMETROS,
    'tamanho_teste': TAMANHO_TESTE,
    'random_state': RANDOM_STATE,
    'num_amostras_treino': len(X_treino),
    'num_amostras_teste': len(X_teste),
    'num_features': X.shape[1]
}

arquivo_config = os.path.join(pasta_execucao, 'configuracao_utilizada.txt')
with open(arquivo_config, 'w', encoding='utf-8') as f:
    f.write("CONFIGURAÇÃO DO EXPERIMENTO\n")
    f.write("="*50 + "\n\n")
    for chave, valor in config_dict.items():
        f.write(f"{chave}: {valor}\n")

    f.write(f"\nRESULTADOS FINAIS\n")
    f.write("-"*30 + "\n")
    f.write(f"R² no teste: {metricas_teste['R2']:.6f}\n")
    f.write(f"RMSE no teste: {metricas_teste['RMSE']:.2f}\n")
    f.write(f"MAE no teste: {metricas_teste['MAE']:.2f}\n")

print(f"⚙️  Configurações salvas: {arquivo_config}")

# Gráfico de importância (só para RandomForest e GradientBoosting)
if MODELO in ['RandomForest', 'GradientBoosting']:
    print(f"\n🌲 Gerando gráfico de importância das features...")
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    top_10 = importancias.nlargest(10)

    plt.figure(figsize=(12, 8))
    top_10.plot(kind='barh', color='skyblue', edgecolor='navy')
    plt.title(f'Top 10 - Importância das Features\nModelo: {MODELO}', fontsize=14, pad=20)
    plt.xlabel('Importância Relativa', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    arquivo_grafico = os.path.join(pasta_execucao, 'importancia_features.png')
    plt.savefig(arquivo_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico salvo: {arquivo_grafico}")
elif MODELO == 'Ridge':
    print(f"\n📈 Para Ridge, a importância está nos coeficientes (não há gráfico automático)")

# ===========================================
# RESUMO FINAL DETALHADO
# ===========================================

print_titulo("🏆 RESUMO FINAL DETALHADO")

print(f"🤖 Modelo: {MODELO}")
print(f"⚙️  Parâmetros:")
for param, valor in PARAMETROS.items():
    print(f"   • {param}: {valor}")

print(f"\n📊 Performance no Teste:")
print(f"   • R²: {metricas_teste['R2']:.4f} ({metricas_teste['R2']*100:.1f}% da variância explicada)")
print(f"   • RMSE: {metricas_teste['RMSE']:,.0f} views")
print(f"   • MAE: {metricas_teste['MAE']:,.0f} views")

print(f"\n🎯 Avaliação Geral:")
if metricas_teste['R2'] >= 0.9:
    print("   🌟 EXCELENTE! Modelo muito preciso")
elif metricas_teste['R2'] >= 0.7:
    print("   👍 MUITO BOM! Modelo confiável")
elif metricas_teste['R2'] >= 0.5:
    print("   🔧 RAZOÁVEL. Pode ser melhorado")
else:
    print("   ❌ FRACO. Precisa de melhorias significativas")

print(f"\n📁 Arquivos salvos em: {pasta_execucao}/")
print(f"   • metricas_detalhadas.csv")
print(f"   • configuracao_utilizada.txt")
if MODELO in ['RandomForest', 'GradientBoosting']:
    print(f"   • importancia_features.png")

print(f"\n📊 Estrutura organizacional:")
print(f"   📂 {pasta_modelo}/ (pasta principal do modelo)")
print(f"   └── 📁 {nome_execucao}/ (esta execução)")

print("\n" + "="*60)
print("✨ EXECUÇÃO CONCLUÍDA COM SUCESSO! ✨")
print("="*60)
