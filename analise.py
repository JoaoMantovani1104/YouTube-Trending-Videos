import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = 'analise_dataset_pre_processado'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Diretório '{OUTPUT_DIR}' criado para salvar os gráficos.")

# CARREGAMENTO E INSPECAO DOS DADOS
df = pd.read_csv('pre_processed_youtube_trending_data.csv')

print("--- 1.1 Inspeção Inicial: Primeiras Linhas ---")
print(df.head())

print("\n--- 1.2 Informações das Colunas (Tipos de Dados e Valores Não-Nulos) ---")
df.info()

# ANALISE DE VALORES AUSENTES E OUTLIERS
print("\n--- 2.1 Valores Ausentes (Missing Values) ---")
print(df.isnull().sum())

df['publish_time'] = pd.to_datetime(df['publish_time'])

# ESTATISTICAS DESCRITIVAS BASICAS
numerical_cols = ['views', 'likes', 'dislikes', 'comment_count']

print("\n--- 3.1 Estatísticas Descritivas para Variáveis Numéricas ---")
stats = df[numerical_cols].describe().loc[['mean', 'std', 'min', 'max']]
print(stats)

# VISUALIZACOES
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribuição das Variáveis Numéricas', fontsize=16)

# HISTOGRAMA - VIEWS
sns.histplot(df['views'], kde=True, ax=axes[0, 0], bins=50)
axes[0, 0].set_title('Distribuição de Views (Variável-Alvo)')
axes[0, 0].ticklabel_format(style='plain', axis='x')

# OUTRAS VARIAVEIS
sns.histplot(df['likes'].dropna(), kde=True, ax=axes[0, 1], bins=50)
axes[0, 1].set_title('Distribuição de Likes')

sns.histplot(df['dislikes'], kde=True, ax=axes[1, 0], bins=50)
axes[1, 0].set_title('Distribuição de Dislikes')

sns.histplot(df['comment_count'].dropna(), kde=True, ax=axes[1, 1], bins=50)
axes[1, 1].set_title('Distribuição de Contagem de Comentários')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '01_histogramas_distribuicao.png'))
plt.close(fig) 

# GRAFICOS DE DISPERCAO
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
plt.savefig(os.path.join(OUTPUT_DIR, '02_dispersao_engajamento.png'))
plt.close(fig)

# HEATMAP
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap de Correlação entre Variáveis Numéricas')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_heatmap_correlacao.png'))
plt.close() 

print(f"\nOs gráficos foram salvos com sucesso no diretório '{OUTPUT_DIR}'.")
