# ==============================================================================
# 5. Pré-processamento e Engenharia de Atributos (Seção 3.3) - Versão Modificada
# ==============================================================================

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Supondo que 'df' é o seu DataFrame carregado e com a coluna 'publish_time' convertida
df = pd.read_csv('youtube_trending_data.csv')
df['publish_time'] = pd.to_datetime(df['publish_time'])


# --- 5.1 Tratamento de Dados Ausentes ---
print("--- 5.1 Tratamento de Dados Ausentes ---")

# Calcular a mediana ANTES de qualquer transformação
median_likes = df['likes'].median()
median_comment_count = df['comment_count'].median()

# Imputar os valores ausentes com a mediana
# NOTA: O inplace=True está gerando um FutureWarning, mas é funcional por agora.
df['likes'].fillna(median_likes, inplace=True)
df['comment_count'].fillna(median_comment_count, inplace=True)
print("Valores ausentes em 'likes' e 'comment_count' foram imputados com a mediana.")


# --- 5.2 Transformação Logarítmica e Padronização ---
# Aplicaremos a transformação log(1+x) e padronização NAS COLUNAS ORIGINAIS.
print("\n--- 5.2 Aplicando Transformação Logarítmica e Padronização ---")
numerical_cols = ['views', 'likes', 'dislikes', 'comment_count']

# 1. Aplicar a Transformação Logarítmica (log(1+x))
for col in numerical_cols:
    df[col] = np.log1p(df[col])
    print(f"Coluna '{col}' transformada com log(1+x).")

# 2. Padronizar as Variáveis Logarítmicas (que agora têm os nomes originais)
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Features numéricas padronizadas com sucesso. Os dados transformados substituíram os originais.")
print("\n--- Amostra das Features Numéricas Após Transformação ---")
print(df[numerical_cols].head())


# --- 5.3 Codificação de Variáveis Categóricas (One-Hot Encoding) ---
print("\n--- 5.3 Aplicando One-Hot Encoding em 'category_id' ---")
# Convertendo 'category_id' para string para evitar tratamento numérico
df['category_id'] = df['category_id'].astype(str)
df_encoded = pd.get_dummies(df, columns=['category_id'], prefix='category', dtype=int)

print(f"Shape do DataFrame antes do One-Hot Encoding: {df.shape}")
print(f"Shape do DataFrame depois do One-Hot Encoding: {df_encoded.shape}")


# ==============================================================================
# 7. Salvando o DataFrame Pré-Processado
# ==============================================================================

# O DataFrame 'df_encoded' agora contém as colunas views, likes, dislikes e
# comment_count com os dados FINALMENTE processados.

original_filename = 'youtube_trending_data.csv'
base_name = original_filename.replace('.csv', '')
output_filename = f'pre_processed_{base_name}.csv'

# Salvar o DataFrame pré-processado
df_encoded.to_csv(output_filename, index=False)

print(f"\n--- Arquivo Salvo ---")
print(f"O DataFrame pré-processado foi salvo como: {output_filename}")
print(f"As colunas numéricas agora contêm os dados transformados e padronizados.")


# ==============================================================================
# O DataFrame 'df_encoded' está pronto para a próxima etapa: Divisão dos Dados
# ==============================================================================