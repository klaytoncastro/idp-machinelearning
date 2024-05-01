import pandas as pd

url = 'https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/decision-tree/winequality-merged.csv'
arquivo = pd.read_csv(url)
arquivo.head()

arquivo['color'] = arquivo['color'].replace('red', 0)
arquivo['color'] = arquivo['color'].replace('white', 1)
arquivo.head()

import numpy as np
arquivo['worst'] = np.where(arquivo['quality'] < 7, 1, 0)
arquivo.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Calcular a matriz de correlação para as colunas numéricas do DataFrame
corr = arquivo.select_dtypes('number').corr()

# Personalizar a paleta de cores (opcional)
custom_palette = sns.color_palette("RdBu_r", n_colors=50)

# Plotar o mapa de calor usando o Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap=custom_palette, fmt=".2f")
plt.title('Mapa de Calor da Matriz de Correlação')
plt.show()

# Calcular a matriz de correlação
correlation_matrix = arquivo.corr()

# Selecionar as correlações com a variável alvo ('quality')
correlation_with_target = correlation_matrix['worst']

# Excluir a correlação da variável alvo consigo mesma
correlation_with_target = correlation_with_target.drop('worst')

# Exibir os valores de correlação e nomes das variáveis preditoras
#print("Correlação com a variável alvo (worst):\n")
#print(correlation_with_target)

# Ordenar os valores de correlação em ordem decrescente
correlation_with_target_sorted = correlation_with_target.sort_values(ascending=False)

# Exibir os valores de correlação e nomes das variáveis preditoras ordenados
print("Correlação com a variável alvo (worst) - Ordenado:\n")
print(correlation_with_target_sorted)

#arquivo = arquivo.drop(['alcohol', free sulfur dioxide', total sulfur dioxide'], axis=1) # Teste removendo várias colunas com baixa correlação

arquivo = arquivo.drop(['quality'], axis=1) #Removendo a coluna 'quality' para não interferir nos cálculos do modelo para predição da variável dependente 'worst'

arquivo.head()

# Exibir o percentual de cada classe
print(arquivo['worst'].value_counts(normalize=True) * 100)

# Definindo os atributos da função de aprendizagem
y = arquivo['worst']
X = arquivo.drop('worst',axis = 1)

from sklearn.model_selection import train_test_split

# Definindo os conjuntos de treino e teste, onde x é o conjunto de atributos (features que são nossas variáveis preditoras) e y é a variável alvo.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=55)

from sklearn.ensemble import ExtraTreesClassifier

#Otimização dos Hiperparâmetros para o Modelo
modelo = ExtraTreesClassifier(n_estimators=40, max_depth=20, min_samples_leaf=1, min_samples_split=2, max_features="sqrt", class_weight="balanced", random_state=42)

#Treinamento e Predição
modelo.fit(x_train, y_train)
y_pred = modelo.predict(x_test)

resultado = modelo.score(x_test, y_test)
print ("Acurácia:", resultado)

from sklearn.metrics import confusion_matrix, classification_report

"""
Calculando e exibindo a matriz de confusão. A orientação padrão é a seguinte:
[0,0]: Verdadeiros Negativos (VN) - Previsões corretamente identificadas como negativas.
[0,1]: Falsos Positivos (FP) - Previsões incorretamente identificadas como positivas.
[1,0]: Falsos Negativos (FN) - Previsões incorretamente identificadas como negativas.
[1,1]: Verdadeiros Positivos (VP) - Previsões corretamente identificadas como positivas.
"""

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

"""
Calculando e exibindo as métricas de classificação.
Se algumas classes têm muito mais amostras do que outras, isso pode influenciar o desempenho e confiabilidade do modelo.

O "support" refere-se à quantidade de ocorrências da classe específica no conjunto de dados, sendo útil para verificar desbalanceamentos.
A "macro avg" calcula a média aritmética das métricas (precisão, recall, F1-score) para cada classe, sem considerar o número de instâncias em cada classe (support).
A "weighted avg" calcula a média ponderada das métricas para cada classe, considerando o número de instâncias em cada classe (support).
"""
print("Relatório de Classsificação:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Prever as probabilidades para os dados de teste.
# Note que estamos interessados nas probabilidades da classe positiva (1), então usamos [:, 1].
y_probs = modelo.predict_proba(x_test)[:, 1]

# Calcular FPR, TPR, e limiares
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calcular a AUC
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from joblib import dump

# Salvar o modelo em um arquivo .pkl
dump(modelo, 'modelo.pkl')

from google.colab import files
files.download('modelo.pkl')

import sklearn
print(sklearn.__version__)

# Selecionar amostras onde a variável worst é igual a 1 (vinho ruim)
ruins = arquivo.query('worst == 1')

# Selecionar amostras onde a variável worst é igual a 0 (vinho bom)
bons = arquivo.query('worst == 0')

bons.head()

ruins.head()

json = bons.to_json(orient='records')

print("\nJSON")
print(json)