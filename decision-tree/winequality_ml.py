# Importando o dataset
import pandas as pd
url = 'https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/decision-tree/winequality-merged.csv'
arquivo = pd.read_csv(url)

# Mostrando as primeiras linhas do DataFrame. 
# print(arquivo.head())
arquivo.head()

# Verificando a coluna 'color' e convertendo os dados categóricos para um formato numérico. 
if 'color' in arquivo.columns:
    print("Column 'color' exists in the DataFrame.")
    # Substituindo valores correspondentes a 'red' por 0, e 'white' por 1
    arquivo['color'] = arquivo['color'].replace('red', 0)
    arquivo['color'] = arquivo['color'].replace('white', 1)
else:
    print("Column 'color' does not exist in the DataFrame.")

# Dividindo os conjuntos de dados para treino e teste.
y = arquivo['color']
X = arquivo.drop('color', axis = 1)

# Importando a funcionalidade de treinamento do modelo 
from sklearn.model_selection import train_test_split

# Considerando x o conjunto de variáveis, y como variável alvo do modelo e definindo o tamanho do conjunto de teste.  
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

# Importando o Algoritmo de Aprendizagem Supervisionada para criação da Árvore de Decisão e treinando o modelo.
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_train, y_train)

# Apresentando a Acurácia do modelo
resultado = modelo.score(x_test, y_test)
print ("Acurácia:", resultado)

# Adicionando a verificação da Precisão, Recall e F1-Score
from sklearn.metrics import precision_score, recall_score, f1_score

# Fazendo previsões no conjunto de teste
y_pred = modelo.predict(x_test)

# Calculando as métricas
precisao = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precisão:", precisao)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.metrics import confusion_matrix, classification_report

"""
# Calculando e exibindo a matriz de confusão. A orientação padrão é a seguinte:
[0,0]: Verdadeiros Negativos (VN) - Previsões corretamente identificadas como negativas.
[0,1]: Falsos Positivos (FP) - Previsões incorretamente identificadas como positivas.
[1,0]: Falsos Negativos (FN) - Previsões incorretamente identificadas como negativas.
[1,1]: Verdadeiros Positivos (VP) - Previsões corretamente identificadas como positivas.
"""

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

"""
# Calculando e exibindo as métricas de classificação. 
Se algumas classes têm muito mais amostras do que outras, isso pode influenciar o desempenho e confiabilidade do modelo. 
O "support" refere-se à quantidade de ocorrências da classe específica no conjunto de dados, sendo útil para verificar desbalanceamentos.  
A "macro avg" calcula a média aritmética das métricas (precisão, recall, F1-score) para cada classe, sem considerar o número de instâncias em cada classe (support). 
A "weighted avg" calcula a média ponderada das métricas para cada classe, considerando o número de instâncias em cada classe (support). 
"""
print("Relatório de Classsificação:")
print(classification_report(y_test, y_pred))