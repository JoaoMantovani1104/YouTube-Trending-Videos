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