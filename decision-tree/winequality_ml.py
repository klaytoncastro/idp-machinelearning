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

print(f'Precisão: {precisao}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')