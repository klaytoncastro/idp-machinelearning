import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
# Substitua 'caminho_do_seu_arquivo.csv' pelo caminho do seu arquivo CSV
df = pd.read_csv('caminho_do_seu_arquivo.csv')

# Visualização geral
print(df.describe())  # Para variáveis numéricas
print(df.info())  # Para tipos de dados e valores ausentes

# Análise descritiva detalhada
# Medidas de tendência central e dispersão
for col in df.columns[:-1]:  # Exclui a coluna 'color' que é categórica
    print(f"{col}:")
    print(f" Média: {df[col].mean()}")
    print(f" Mediana: {df[col].median()}")
    print(f" Desvio Padrão: {df[col].std()}")
    print(f" Mínimo: {df[col].min()}")
    print(f" Máximo: {df[col].max()}\n")

# Contagem de valores para a variável 'color'
print(df['color'].value_counts())

# Histogramas para distribuição de cada variável numérica
df.drop('color', axis=1).hist(figsize=(14, 12), bins=15)
plt.tight_layout()
plt.show()

# Boxplots para identificar outliers
plt.figure(figsize=(10, 8))
sns.boxplot(data=df.drop('color', axis=1))
plt.xticks(rotation=45)
plt.show()

# Relação com a variável alvo (cor)
# Gráficos de violino para cada variável numérica por 'color'
for col in df.columns[:-1]:  # Exclui a coluna 'color'
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='color', y=col, data=df)
    plt.title(f'Relação entre {col} e a Cor do Vinho')
    plt.show()
