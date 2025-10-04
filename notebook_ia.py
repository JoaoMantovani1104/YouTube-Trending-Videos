import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. Carregamento e Inspeção dos Dados
# Substitua 'youtube_trending_data.csv' pelo nome do seu arquivo real
# ==============================================================================

df = pd.read_csv('youtube_trending_data.csv')

print("--- 1.1 Inspeção Inicial: Primeiras Linhas ---")
print(df.head())

print("\n--- 1.2 Informações das Colunas (Tipos de Dados e Valores Não-Nulos) ---")
df.info()

# ==============================================================================
# 2. Análise de Valores Ausentes e Outliers
# ==============================================================================

print("\n--- 2.1 Valores Ausentes (Missing Values) ---")
print(df.isnull().sum())

# Converter 'publish_time' para o formato datetime para análise temporal, se necessário
df['publish_time'] = pd.to_datetime(df['publish_time'])

# ==============================================================================
# 3. Estatísticas Descritivas Básicas
# Variáveis numéricas de engajamento e a variável-alvo 'views'
# ==============================================================================

numerical_cols = ['views', 'likes', 'dislikes', 'comment_count']

print("\n--- 3.1 Estatísticas Descritivas para Variáveis Numéricas ---")
# Calculando a média, desvio-padrão, mínimo e máximo
stats = df[numerical_cols].describe().loc[['mean', 'std', 'min', 'max']]
print(stats)

# ==============================================================================
# 4. Visualizações
# ==============================================================================

# 4.1 Histograma da Variável Alvo 'views' e outras features numéricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribuição das Variáveis Numéricas', fontsize=16)

# Histograma da Variável-Alvo
sns.histplot(df['views'], kde=True, ax=axes[0, 0], bins=50)
axes[0, 0].set_title('Distribuição de Views (Variável-Alvo)')
axes[0, 0].ticklabel_format(style='plain', axis='x')

# Outras variáveis de engajamento
sns.histplot(df['likes'].dropna(), kde=True, ax=axes[0, 1], bins=50)
axes[0, 1].set_title('Distribuição de Likes')

sns.histplot(df['dislikes'], kde=True, ax=axes[1, 0], bins=50)
axes[1, 0].set_title('Distribuição de Dislikes')

sns.histplot(df['comment_count'].dropna(), kde=True, ax=axes[1, 1], bins=50)
axes[1, 1].set_title('Distribuição de Contagem de Comentários')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() # Use plt.show() em um notebook real

# 4.2 Gráficos de Dispersão (Correlações) com a Variável-Alvo 'views'
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Relação entre Engajamento e Views', fontsize=16)

# Views vs. Likes
sns.scatterplot(x='likes', y='views', data=df, ax=axes[0])
axes[0].set_title('Views vs. Likes')
axes[0].ticklabel_format(style='plain', axis='both')

# Views vs. Dislikes
sns.scatterplot(x='dislikes', y='views', data=df, ax=axes[1])
axes[1].set_title('Views vs. Dislikes')
axes[1].ticklabel_format(style='plain', axis='both')

# Views vs. Comment Count
sns.scatterplot(x='comment_count', y='views', data=df, ax=axes[2])
axes[2].set_title('Views vs. Comment Count')
axes[2].ticklabel_format(style='plain', axis='both')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() # Use plt.show() em um notebook real

# 4.3 Heatmap de Correlação
# O Heatmap deve ser feito após a remoção de valores ausentes (se for o caso)
# ou usando um método que lide com eles (como o .corr() padrão que ignora NaN)
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap de Correlação entre Variáveis Numéricas')
plt.tight_layout()
plt.show() # Use plt.show() em um notebook real

##Segunda etapa, pré-processamento


# ==============================================================================
# 5. Pré-processamento e Engenharia de Atributos (Seção 3.3)
# Implementação baseada na metodologia descrita no relatório.
# ==============================================================================

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Supondo que 'df' é o seu DataFrame carregado e com a coluna 'publish_time' convertida
# df = pd.read_csv('youtube_trending_data.csv')
# df['publish_time'] = pd.to_datetime(df['publish_time'])


# --- 5.1 Tratamento de Dados Ausentes ---
# Conforme o relatório, os valores nulos em 'likes' e 'comment_count'
# [cite_start]serão imputados com a mediana para minimizar a influência de outliers[cite: 151].

print("--- 5.1 Tratamento de Dados Ausentes ---")
print(f"Valores ausentes em 'likes' antes: {df['likes'].isnull().sum()}")
print(f"Valores ausentes em 'comment_count' antes: {df['comment_count'].isnull().sum()}")

# Calcular a mediana ANTES de qualquer transformação
median_likes = df['likes'].median()
median_comment_count = df['comment_count'].median()

# Imputar os valores ausentes com a mediana
df['likes'].fillna(median_likes, inplace=True)
df['comment_count'].fillna(median_comment_count, inplace=True)

print(f"\nValores ausentes em 'likes' depois: {df['likes'].isnull().sum()}")
print(f"Valores ausentes em 'comment_count' depois: {df['comment_count'].isnull().sum()}\n")


# --- 5.2 Transformação Logarítmica ---
# Aplicar a transformação log(1+x) nas variáveis com forte assimetria para
# [cite_start]estabilizar a variância, conforme planejado[cite: 152].
# Usamos np.log1p(x) que é equivalente a np.log(1+x) e lida com valores 0 de forma segura.

print("--- 5.2 Aplicando Transformação Logarítmica ---")
cols_to_log = ['views', 'likes', 'comment_count']
# A coluna 'dislikes' também possui assimetria, então incluí-la na transformação é uma boa prática.
if 'dislikes' not in cols_to_log:
    cols_to_log.append('dislikes')

for col in cols_to_log:
    df[f'{col}_log'] = np.log1p(df[col])
    print(f"Coluna '{col}_log' criada.")

# O DataFrame agora contém as versões originais e as transformadas.
# Para a modelagem, usaremos as colunas com sufixo '_log'.


# --- 5.3 Codificação de Variáveis Categóricas (One-Hot Encoding) ---
# [cite_start]A variável 'category_id' será transformada usando One-Hot Encoding[cite: 153].
# Isso cria novas colunas binárias para cada categoria.

print("\n--- 5.3 Aplicando One-Hot Encoding em 'category_id' ---")
# Usando pd.get_dummies para aplicar o One-Hot Encoding
# O prefixo 'category' ajuda a identificar as novas colunas
# Convertendo 'category_id' para string para evitar tratamento numérico
df_encoded = pd.get_dummies(df, columns=['category_id'], prefix='category', dtype=int)

print("Shape do DataFrame antes do One-Hot Encoding:", df.shape)
print("Shape do DataFrame depois do One-Hot Encoding:", df_encoded.shape)
print("Novas colunas de categoria criadas.")


# --- 5.4 Normalização (Padronização) ---
# As variáveis numéricas transformadas serão padronizadas para terem média 0 e desvio padrão 1,
# [cite_start]garantindo que contribuam igualmente para o modelo[cite: 154].

print("\n--- 5.4 Padronizando as Features Numéricas ---")
scaler = StandardScaler()

# Selecionar as colunas numéricas que foram transformadas para a padronização
numerical_log_cols = ['views_log', 'likes_log', 'dislikes_log', 'comment_count_log']

# 'Fitar' e transformar os dados. O resultado é um array NumPy.
df_encoded[numerical_log_cols] = scaler.fit_transform(df_encoded[numerical_log_cols])

print("Features numéricas padronizadas com sucesso.")
print("\n--- Amostra dos dados após pré-processamento ---")
# Exibindo as colunas transformadas para verificação
print(df_encoded[numerical_log_cols].head())


# ==============================================================================
# O DataFrame 'df_encoded' está pronto para a próxima etapa: Divisão dos Dados
# ==============================================================================

