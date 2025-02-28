#pip install pandas

# Importando o dataset
import pandas as pd

url = 'https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/decision-tree/winequality-merged.csv'
arquivo = pd.read_csv(url)
arquivo.head()

# Informações gerais sobre o dataset
print(arquivo.info())

# Descrição estatística das variáveis numéricas
print(arquivo.describe())

# Verificando a quantidade de cada tipo nas amostras 
print(arquivo['color'].value_counts())

#pip install matplotlib

# Gerando histogramas
import matplotlib.pyplot as plt
arquivo.hist(figsize=(20, 15), bins=20)
plt.show()

#Gerando boxplots
arquivo.boxplot(figsize=(20, 10), rot=90)
plt.show()

# Transformando a variável categórica em numérica

pd.set_option('future.no_silent_downcasting', True) # Configuração do modeo de downcast

if 'color' in arquivo.columns:
    print("Column 'color' exists in the DataFrame.")
    # Replace 'red' with 0 and 'white' with 1
    arquivo['color'] = arquivo['color'].replace('red', 0)
    arquivo['color'] = arquivo['color'].replace('white', 1)
else:
    print("Column 'color' does not exist in the DataFrame.")

# Verificando a transformação. 
print(arquivo.head())

# Dropando a coluna da variável alvo. 

#y = arquivo['color']
#X = arquivo.drop('color', axis = 1)

y = arquivo['quality']
X = arquivo.drop('quality', axis = 1)

# Definindo X como features e y como variável alvo e efetuando a divisão dos dados para treinamento e teste

# pip install scikit-learn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importando StandardScaler
from sklearn.preprocessing import StandardScaler

# Escalamento dos dados
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# RFE
modelo_rfe = ExtraTreesRegressor()
rfe = RFE(estimator=modelo_rfe, n_features_to_select=10)  # Ajustar conforme necessário
x_train_rfe = rfe.fit_transform(x_train_scaled, y_train)
x_test_rfe = rfe.transform(x_test_scaled)

# PCA
pca = PCA(n_components=0.95)  # Mantém 95% da variancia
x_train_pca = pca.fit_transform(x_train_rfe)
x_test_pca = pca.transform(x_test_rfe)
"""

# Definindo o algoritimo e treinando o modelo

from sklearn.ensemble import ExtraTreesRegressor
modelo = ExtraTreesRegressor()

modelo.fit(x_train, y_train)

# Apresentando o resultado

resultado = modelo.score(x_test, y_test)
print ("Acurácia:", resultado)

#### Verificando as métricas base 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Realizando previsões com o modelo
y_pred = modelo.predict(x_test)  

# Calculando as métricas, onde y_pred contém as previsões e y_test os valores reais
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimindo as métricas
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)

"""
Acurácia: 0.5391484512911533

MSE : 1.2343589743589745
RMSE : 1.1110170900391112
MAE : 0.8261538461538461
R² : -0.6161209584924814
"""

# Arredondamento das previsões para o inteiro mais próximo
import numpy as np
y_pred_arredondado = np.round(y_pred)

# Ajuste dos valores previstos para garantir que estejam dentro da escala de inteiros de 1 a 10
y_pred_ajustado = np.clip(y_pred_arredondado, 1, 10)

# Recálculo das métricas usando os valores previstos ajustados
mse = mean_squared_error(y_test, y_pred_ajustado)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_ajustado)
r2 = r2_score(y_test, y_pred_ajustado)

# Reimprimindo as métricas ajustadas
print("MSE ajustado:", mse)
print("RMSE ajustado:", rmse)
print("MAE ajustado:", mae)
print("R² ajustado:", r2)

# Calculando a diferença absoluta entre as previsões ajustadas e os valores reais
diferencas_absolutas = np.abs(y_pred_ajustado - y_test)

# Definindo um critério de acurácia (previsões dentro de ±1 unidade do valor real)
criterio_acuracia = 1

# Calculando a proporção de previsões que satisfazem o critério
acuracia_ajustada = np.mean(diferencas_absolutas <= criterio_acuracia)

# Exibindo a acurácia ajustada
print("Acurácia ajustada (previsões dentro de ±1):", acuracia_ajustada)

"""
Acurácia ajustada:
(proporção de previsões dentro de ±1 da real):
0.966153846153846

MSE ajustado: 0.4323076923076923
RMSE ajustado: 0.6575010968110184
MAE ajustado: 0.35025641025641024
R² ajustado: 0.453820531776159
"""