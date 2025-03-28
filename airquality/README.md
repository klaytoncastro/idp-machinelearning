# Elaborando uma abordagem de Machine Learning para uma Tarefa de Regressão

Nessa tarefa, apresente uma abordagem de análise exploratória, pré-processamento, modelagem, otimização e interpretação para fornecer uma visão sobre os fatores que influenciam a qualidade do ar e como diferentes variáveis podem ser usadas para prever a concentração de poluentes.

## Descrição do Dataset

O dataset pode ser encontrado no seguinte link: [Air Quality Dataset](https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/datasets/AirQualityUCI.csv). Ele contém informações sobre a qualidade do ar em uma região específica, medindo várias características como concentração de poluentes e outras variáveis ambientais. Os atributos (features) incluem:

<!--https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip-->

- **Date**: Data da coleta.
- **Time**: Hora da coleta.
- **CO(GT)**: Concentração de monóxido de carbono em mg/m^3.
- **PT08.S1(CO)**: Indicador de sensor da concentração de CO.
- **NMHC(GT)**: Hidrocarbonetos não metânicos em microg/m^3.
- **C6H6(GT)**: Concentração de benzeno em microg/m^3.
- **PT08.S2(NMHC)**: Indicador de sensor da concentração de NMHC.
- **NOx(GT)**: Concentração de óxidos de nitrogênio em ppb.
- **PT08.S3(NOx)**: Indicador de sensor da concentração de NOx.
- **NO2(GT)**: Concentração de dióxido de nitrogênio em microg/m^3.
- **PT08.S4(NO2)**: Indicador de sensor da concentração de NO2.
- **O3(GT)**: Concentração de ozônio em microg/m^3.
- **PT08.S5(O3)**: Indicador de sensor da concentração de O3.
- **T**: Temperatura em °C.
- **RH**: Umidade relativa em %.
- **AH**: Umidade absoluta em g/m^3.

## Definição do Problema 

Como Cientista de Dados / Engenheiro de Machine Learning, desenvolva um modelo que ajude a entender como diferentes fatores ambientais influenciam os níveis de monóxido de carbono na atmosfera. 

**Dica:** Uma abordagem possível seria prever a concentração de CO (CO(GT)) com base nas outras variáveis disponíveis no dataset. 

## Análise Exploratória de Dados (EDA)

1. **Visualização Inicial**: Explore a distribuição das variáveis com histogramas e boxplots.
2. **Correlação**: Utilize mapas de calor para identificar correlações entre as variáveis.
3. **Tendências Temporais**: Analise como as variáveis variam ao longo do tempo.

## Pré-processamento de Dados

1. **Tratamento de Valores Ausentes**: Verifique e trate valores ausentes. Pode-se optar por imputação ou remoção de linhas com valores ausentes.
2. **Conversão de Tipos de Dados**: Certifique-se de que todas as variáveis estejam corretamente tipadas no dataframe, inclusive datas. 
3. **Normalização/Padronização**: Normalize ou padronize as variáveis numéricas para garantir que todas estejam na mesma escala.

<!--

(e.g., datas como datetime)

-->

## Seleção de Features e Modelagem

- **Seleção de Features**: Use técnicas como seleção de features baseada em sua importância para ajustar seu dataset. 

<!-- ou métodos automáticos como RFE (Recursive Feature Elimination). -->

- **Algoritmos de Regressão**:
  - Regressão Linear
  - Random Forest Regressor
  - Extra Trees Regressor
  - Gradient Boosting Regressor
  - K-Nearest Neighbors Regressor
  - Support Vector Machine Regressor (SVM / SVR)

Utilize validação cruzada para avaliar a robustez do modelo.

## Otimização de Hiperparâmetros

- Utilize métodos como Random Search e Grid Search para encontrar os melhores hiperparâmetros para os algoritmos que desempenharem melhor a tarefa.  

## Avaliação e Interpretação do Modelo

1. **Avaliação do Modelo**: Use métricas como Mean Squared Error (MSE), Mean Absolute Error (MAE) e R² para avaliar o desempenho do modelo.
2. **Importância das Features**: Analise a importância das features para entender quais variáveis têm maior influência na predição da concentração de CO.
3. **Resíduos do Modelo**: Examine os resíduos (diferença entre valores preditos e reais) para identificar padrões ou anomalias.

<!--
```python
# -*- coding: utf-8 -*-
Solução
"""
# Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
from sklearn.ensemble import IsolationForest, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregando o Dataset
url = 'https://raw.githubusercontent.com/klaytoncastro/idp-storytelling/master/airquality/airquality.csv'
df = pd.read_csv(url, delimiter = ';', decimal = ',')
df.head()

# Verificando a estrutura de dados
df.info()

# Convertendo as colunas Date e Time para DateTime
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')

# Removendo as colunas originais Date e Time
df.drop(columns=['Date', 'Time'], inplace=True)
df.head()

df.info()

df.head()

# Calculando as correlações entre as variáveis ​​preditoras e a variável alvo
correlations = df.corr()['CO(GT)'].sort_values(ascending=False)
print(correlations)

# Verificando a distribuição dos dados
df.describe()

# Identificamos um padrão estranho, onde -200 aparece como valor mínimo para cada uma das variáveis.
# Por isso, vamos contar valores -200 em cada coluna e avaliar se isso é frequente ou eventual.
print("Contagem de valores -200 em cada coluna:")
for column in df.columns:
    count_negative_200 = (df[column] == -200).sum()
    print(f"{column}: {count_negative_200}")

# De fato, são valores anômalos. Vamos substituir -200 por NaN (NULL)
df.replace(-200, np.nan, inplace=True)
df.describe()

# Verificando quantidade de missing values por coluna
missing_values = df.isna().sum().div(df.shape[0]).to_frame().sort_values(by=0, ascending=False)
missing_values.plot(kind='bar', figsize=(10, 5))
plt.title('Porcentagem de valores ausentes por coluna')
plt.show()

# Decidimos a descartar a coluna NMHC(GT), mais 80% de valores ausentes. Imputar a mediana pode apresentar padrões lineares artificiais.
df.drop(columns=['NMHC(GT)'], inplace=True)
df.describe()

# Criando uma nova coluna para agregar os valores e gerando um gráfico de barras dos total de valores ausentes por registro único
df['missing_values'] = df.isnull().any(axis=1)
df.groupby('missing_values').size().plot(kind='bar')
plt.title('Número de valores ausentes por observação')
plt.show()

df.head()

# Hipótese 1: se descartarmos as demais colunas ou registros com valores nulos, perderemos muita capacidade de previsão do modelo.
# Poderiamos preencher os valores NaN com a mediana da coluna, mas os padrões lineares gerados seriam de fato artificiais.
# Dessa forma, vamos seguir com a Hipótese 2.
'''
for column in df.columns:
    if df[column].isnull().any():
        df[column].fillna(df[column].median(), inplace=True)

df.describe()
'''

# Hipótese 2: Como ainda teremos em torno de 7000 observações na amostra após remover os dados ausentes, decidimos removê-los para assegurar maior fidelidade.
df = df.dropna()
df.describe()

# As distribuições parecem melhores agora. Vamos exibir a nova matriz de correlação para análise.

correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin = -1)
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()

# Vamos excluir a coluna intermediária 'missing_values' e a coluna 'DateTime'
df.drop(columns=['DateTime'], inplace=True)
#df.drop(columns=['missing_values'], inplace=True)

# Vamos manter as demais variáveis de baixa correlação por enquanto.
df.drop(columns=['T'], inplace=True)
df.drop(columns=['RH'], inplace=True)
df.drop(columns=['AH'], inplace=True)

df.head()

# Calculando as correlações entre as variáveis ​​preditoras e a variável alvo
correlations = df.corr()['CO(GT)'].sort_values(ascending=False)
print(correlations)

# Preparando as variáveis para treinar o modelo.
X = df.drop('CO(GT)', axis=1)
y = df['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar com o ExtraTrees
#model = ExtraTreesRegressor(random_state=7, n_estimators=67, max_features='sqrt', max_depth=100, min_samples_split=13, min_samples_leaf=1, bootstrap = False)
#model = ExtraTreesRegressor(random_state=42, n_estimators=350, max_features='sqrt', max_depth=None, min_samples_split=2, min_samples_leaf=1)
model = ExtraTreesRegressor();
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R² Score: {r2}')

df.describe()

# Gráfico de valores previstos x valores atuais
plt.figure(figsize=(10, 6))
plt.scatter(x=y_test, y=y_pred, color='blue', label='Valores Previstos')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='-', label='Ideal Line')
plt.xlabel('Valores reais')
plt.ylabel('Valores Previstos')
plt.title('Valores previstos x Valores reais')
plt.legend()
plt.show()
```
-->