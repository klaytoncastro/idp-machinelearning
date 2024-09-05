# Elaborando uma Abordagem de Machine Learning para uma Tarefa de Classificação e Clusterização

## Descrição do Dataset

O dataset **Iris** é um conjunto de dados clássico usado tanto para tarefas de classificação quanto para clusterização, contendo informações sobre três diferentes espécies de flores do gênero *Iris*: **Iris-setosa**, **Iris-versicolor** e **Iris-virginica**. O objetivo é classificar ou agrupar as flores com base nas suas características morfológicas.

Você pode acessar o dataset diretamente do **scikit-learn** ou baixá-lo utilizando o link a seguir:

[Dataset Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)

### Atributos do Dataset:

O dataset contém 150 amostras de flores, com 4 características numéricas (features):

- **SepalLength**: Comprimento da sépala (em centímetros).
- **SepalWidth**: Largura da sépala (em centímetros).
- **PetalLength**: Comprimento da pétala (em centímetros).
- **PetalWidth**: Largura da pétala (em centímetros).
- **Species**: Classe de cada flor (*Iris-setosa*, *Iris-versicolor* ou *Iris-virginica*).

## Tarefa

O objetivo é que vocês implementem tanto **classificação** quanto **clusterização** no dataset **Iris**. 

### Variável Alvo: Species

Utilize os notebooks apresentados em sala de aula como referência para o fluxo de trabalho da abordagem de clusterização e classificação. Clique no link abaixo para baixar os notebooks Mall Customers e Wine Quality:

- [Mall Customers](https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/mall/cluster_mall.ipynb)
- [Wine Quality](https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/decision-tree/winequality_ml_classifier.ipynb)

### Etapas a Seguir:

### 1. Análise Exploratória de Dados (EDA)

- **Visualização Inicial**: Explore o dataset visualmente usando gráficos de dispersão para verificar como as espécies de flores se distribuem em relação às suas características (sépalas e pétalas).
- **Distribuição das Classes**: Verifique a distribuição das diferentes espécies de flores para garantir que o dataset esteja balanceado.
- **Correlação**: Explore a relação entre as variáveis de características (sépalas e pétalas) e suas possíveis correlações.

### 2. Pré-processamento de Dados

1. **Tratamento de Valores Ausentes**: Verifique se há valores ausentes (caso o dataset seja carregado de forma diferente) e trate-os adequadamente.
2. **Normalização/Padronização**: Normalize as features numéricas para garantir que todas as variáveis estejam na mesma escala, o que é importante para alguns algoritmos de machine learning.
3. **Divisão de Dados**: Separe o dataset em conjuntos de treino e teste para validação dos modelos de classificação.

### 3. Modelagem: Classificação

Aplique diferentes algoritmos de **classificação** para prever a espécie das flores, como:

- **K-Nearest Neighbors (KNN)**: Comece com o KNN para identificar as espécies de flores com base nos vizinhos mais próximos.
- **Logistic Regression**: Um algoritmo simples e eficiente para classificação binária ou multiclasses.
- **Support Vector Machines (SVM)**: Útil para problemas de classificação linear e não-linear.
- **Árvores de Decisão e Random Forest**: Capturam interações complexas entre as variáveis e podem ser comparados para melhorar a performance.
- **Gradient Boosting Machines (GBM)**: Use modelos como XGBoost para melhorar a precisão e reduzir o overfitting.

### 4. Modelagem: Clusterização

Aplique diferentes algoritmos de **clusterização** para agrupar as flores em três clusters sem usar as classes como rótulo:

- **K-Means**: Aplique o algoritmo de K-Means para separar as flores em 3 clusters e analise se os clusters correspondem às espécies reais.
- **DBSCAN**: Um algoritmo de clusterização baseado em densidade que lida bem com outliers.
- **Hierarchical Clustering (Agglomerative Clustering)**: Visualize como as flores são agrupadas hierarquicamente.

### 5. Avaliação dos Modelos

- **Clusterização**: Avalie os clusters formados utilizando métricas como:
  - **Índice de Silhueta**: Para medir o quão bem os pontos estão agrupados em cada cluster.
  - **Visualização de Clusters**: Use gráficos 2D para visualizar os clusters formados e comparar com as classes reais.

- **Classificação**: Avalie os modelos de classificação usando métricas como:
  - **Acurácia**: Para medir a precisão do modelo em prever as espécies corretas.
  - **Relatório de Classificação**: Use métricas como precisão, recall e F1-score.
  - **Matriz de Confusão**: Visualize os erros de classificação.

### 6. Interpretação e Visualização

- **Visualização dos Clusters**: Gere gráficos de dispersão para visualizar os clusters formados e comparar com os valores reais das classes.
- **Análise de Desempenho**: Compare os resultados de clusterização com a classificação supervisionada. Quão bem o algoritmo de clusterização conseguiu separar as espécies de flores?

### Conclusão

Com base nos modelos e nas métricas utilizadas, discuta as principais descobertas. Em quais aspectos a clusterização foi bem-sucedida? Quais algoritmos de classificação apresentaram o melhor desempenho?
