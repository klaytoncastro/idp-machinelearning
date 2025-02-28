# Otimização de Hiperparâmetros em Machine Learning

## Objetivo

Nesta tarefa, cada estudante ou dupla deverá implementar os três métodos de otimização a seguir, buscando o melhor desempenho possível do algoritmo atribuído:

1. **Grid Search**: Realiza uma busca exaustiva testando todas as combinações possíveis de hiperparâmetros dentro de uma grade predefinida.
2. **Randomized Search**: Realiza uma busca aleatória, explorando combinações de hiperparâmetros de forma mais eficiente que o Grid Search para espaços de busca maiores.
3. **Otimização Bayesiana**: Utiliza modelos probabilísticos para guiar a busca pelos melhores hiperparâmetros, aproveitando informações de iterações anteriores para realizar uma busca mais inteligente.

Cada estudante ou dupla deverá otimizar um modelo de machine learning utilizando os três métodos clássicos de otimização de hiperparâmetros apresentados em sala de aula: **Grid Search**, **Randomized Search** e **Otimização Bayesiana**. O objetivo é comparar o desempenho dos modelos com diferentes configurações de hiperparâmetros e avaliar o impacto desta parametrização nas métricas de performance.  

## Algoritmos

Cada estudante ou dupla será responsável por um dos seguintes algoritmos, que foram previamente sorteados em sala de aula, e deverá(ão) realizar a otimização de seus respectivos hiperparâmetros na tarefa de classificação de vinhos do dataset [Wine Quality](https://github.com/klaytoncastro/idp-machinelearning/blob/main/optimization/winequality_ml_classifier_optimized.ipynb):

# Alocações dos Trabalhos em Dupla

### Decision Tree
- Felipe Pereira Dutra
- Kelwin dos Santos Menezes

### Random Forest
- Sara Pacheco de Azevedo
- Fábio Luís de Carvalho Terra

### Extra Tree
- Pedro Calil Raposo Mingossi Cordeiro
- João Gabriel Gonçalves Oliveira

### Extra Trees
- Arthur Torquato Novaes
- Felipe Barroso de Castro

### XGBoost
- Eduardo Milagres Lima
- Igor Caldeira Andrade

### LightGBM
- Luca Verdade Lenzoni
- Lucas Fiche Ungarelli Borges

### Naive Bayes
- Claudio da Aparecida Meireles Filho
- Pedro Rodrigues de Araújo

### Logistic Regression
- Mariana Magalhaes Silva
- Leonardo Freitas Barboza

### SVM (Support Vector Classifier - SVC)
- Lucas Fidalgo Bitencourt
- Mateus Batista Peixoto da Silva

### k-NN (k-Nearest Neighbors)
- João Henrique de Oliveira Salles
- Lucas Narita Nunes de Melo Freita


## Técnicas de Otimização



## Recursos Necessários

- **Slides e Notebooks de Exemplo**: Disponíveis no repositório [Notas de Aula](https://github.com/klaytoncastro/idp-machinelearning/blob/main/optimization/ML_Optimization.pdf) e nos arquivos `.ipynb` de otimização.
- **Documentação do scikit-learn**: [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- **Documentação do XGBoost**: [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- **Documentação do LightGBM**: [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)

## Métricas de Avaliação

Os modelos devem ser avaliados utilizando as seguintes métricas:

- **Acurácia**: Percentual de previsões corretas sobre o total de previsões.
- **Precisão**: Proporção de verdadeiros positivos sobre o total de positivos previstos.
- **Recall**: Proporção de verdadeiros positivos sobre o total de positivos reais.
- **F1-Score**: Média harmônica entre precisão e recall, equilibrando ambos.

## Formato do Relatório
Cada estudante ou dupla deverá documentar o processo de otimização, com as seguintes seções:

1. **Introdução ao Algoritmo**: Descrição do algoritmo utilizado e de seus principais hiperparâmetros.
2. **Metodologia**: Explicação das técnicas de otimização e hiperparâmetros testados. Justifique a escolha das métricas usadas para otimização.
3. **Resultados**: Apresente uma comparação dos resultados das três técnicas de otimização e seus respectivos desempenhos. Inclua gráficos, tabelas e discussões sobre os tempos de execução e a efetividade de cada método.
4. **Discussão**: Reflexões sobre os resultados, com possíveis melhorias para otimização e sobre o impacto da escolha dos hiperparâmetros.

## Apresentação

Os resultados deverão ser apresentados em sala de aula, com o uso de gráficos e tabelas que ilustrem o impacto das diferentes técnicas de otimização sobre o desempenho do modelo. 


<!--

## Tarefa 01: 

- Após **executar os notebooks passo a passo**, **entender o que o código está realizando**, responda: 

a) Quais células precisam ser ajustadas no notebook da tarefa de classificação? Por que? 

b) Quais células precisam ser ajustadas no notebook da tarefa de regressão? Por que? 

- Agora prossiga para as tarefas abaixo, onde o encorajamos a **explorar outros algoritmos** e **avaliar o seu desempenho**. 

## Tarefa 02: 

- Teste outros algoritmos para tarefas de classificação (color) e regressão (quality), conforme a pesquisa em grupo apresentada em sala de aula. 

- Para problemas de classificação, além do algoritmo `ExtraTreesClassifier`, que faz uma robusta implementação baseada em Árvore de Decisão, `Naive Bayes` e `Support Vector Machine (SVM)` são alternativas populares, dependendo da natureza dos dados e do problema específico que você está tentando resolver. 

### Usando Naive Bayes para Classificação

O Naive Bayes é uma técnica de classificação baseada em aplicar o teorema de Bayes com a "ingenuidade" de supor independência entre os preditores. É fácil de construir e particularmente útil para grandes volumes de dados. Além disso, é eficaz em problemas de classificação multinomial e binomial. 

Existem diferentes implementações de Naive Bayes no Scikit-Learn, adequados para diferentes tipos de dados:

- GaussianNB: Usado em classificação onde as features são contínuas e seguem uma distribuição normal.
- MultinomialNB: Bom para quando suas features são contagens ou frequências de termos (comumente usado em classificação de texto).
- BernoulliNB: Adequado para features binárias.

```python

from sklearn.naive_bayes import GaussianNB

# Para dados com features contínuas que seguem uma distribuição aproximadamente normal
modelo_nb = GaussianNB()

modelo_nb.fit(X_train, y_train)
y_pred = modelo_nb.predict(X_test)

```

### Usando SVM para Classificação

O Support Vector Machine (SVM) é um método poderoso e versátil para tarefas de classificação e detecção de outliers. Para classificação, especialmente em casos de categorias claramente distintas, o SVM pode ser eficaz. O `Scikit-Learn` oferece várias implementações do SVM, incluindo SVC (Support Vector Classification), que é mais comumente usado para problemas de classificação. Teste e avalie os resultados: 

```python
from sklearn.svm import SVC
# Inicializando o classificador SVM com um kernel. O padrão é 'rbf', mas pode ser alterado para 'linear', 'poly', etc.
modelo_svm = SVC(kernel='linear')

modelo_svm.fit(X_train, y_train)
y_pred = modelo_svm.predict(X_test)
```

### Usando Regressão Logística

Embora seja chamada de regressão, esta técnica é utilizada para classificação binária (prever entre duas classes). Estima probabilidades usando uma função logística que mapeia qualquer valor real para um valor entre 0 e 1. É ideal para problemas onde a variável dependente é categórica (por exemplo, sim/não, verdadeiro/falso).

```python
from sklearn.linear_model import LogisticRegression

# Criando uma instância do modelo
model = LogisticRegression()

# Treinando o modelo com os dados de treino
model.fit(X_train, y_train)
```

### Avaliação dos Modelos

Após treinar o seu modelo, você precisa avaliar o quão bem ele performa. Para classificação, vimos que métricas comuns incluem acurácia, precisão, recall, e a F1-score. O Scikit-Learn fornece funções prontas para calcular essas métricas:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
```

- Qual algoritmo performou melhor? 

- Qual algoritmo foi mais rápido? 

## Tarefa 03



### Support Vector Machine (SVM) para Regressão (SVR)

O SVM não serve apenas para classificação. O Support Vector Regression (SVR) é a versão do SVM usada para problemas de regressão. O SVR pode ser eficaz em espaços de alta dimensão e em casos onde o número de dimensões excede o número de amostras.

```python
from sklearn.svm import SVR

modelo_svr = SVR(kernel='linear') # Você pode experimentar com diferentes kernels como 'linear', 'poly', 'rbf'.
```

### Regressão Linear

Um dos métodos mais simples e amplamente usados. Bom ponto de partida para problemas de regressão devido à sua simplicidade e interpretabilidade.

```python
from sklearn.linear_model import LinearRegression

modelo_lr = LinearRegression()
```

### Regressão Ridge

```python
from sklearn.linear_model import Ridge

modelo_ridge = Ridge(alpha=1.0) # O parâmetro alpha controla a força da regularização.
```

### Regressão Lasso

```python
from sklearn.linear_model import Lasso

modelo_lasso = Lasso(alpha=0.1)
```

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

modelo_rfr = RandomForestRegressor()
```

### Avaliação dos Modelos

Ao invés de Precision, Recall, F1-Score, que são métricas adequadas para tarefas de classificação, utilize MSE, RMSE, MAE e coeficiente R2, que são adequadas a um problema de regressão e verifique o desempenho de seu modelo. 

```python
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
```

- Agora responda: 

a) Qual algoritmo performou melhor? Por que? 

b) Qual algoritmo foi executado mais rápido e qual foi executado com maior dificuldade? 

c) Como podemos conciliar o impacto entre o tempo de execução e o custo benefício para obter performance? 

## Tarefa 04

- Chegou a hora de refinar seu modelo ML de forma iterativa. Observe a relação com a variável alvo: explore como as variáveis se relacionam. 

a) Qual o impacto dos outliers? 

b) Quais variáveis podem ser removidas no modelo para previsão da qualidade? 

c) Após remover essas variáveis (utilize o Pandas para isso), como se ajusta o modelo e se comportam as métricas de avaliação? 

d) Otimize o desempenho utilizando como base as métricas obtidas na tarefa de regressão, onde há maior margem para otimização. 

- Os estudantes que conseguirem realizar uma otimização e explicar com sucesso o trabalho realizado receberão pontuação extra! 

| **Métrica** | **Valor**          |
|---------|------------------------|
| Acurácia| 0.54281651011084       |
| MSE     | 0.3498934358974359     |
| RMSE    | 0.5915179083488816     |
| MAE     | 0.39295384615384615    |
| R²      | 0.54281651011084       |
| **Pós-Arrendondamento**          |
| Acurácia ajustada | 0.9702564102564103 | 
| MSE ajustado | 0.4266666666666667 | 
| RMSE ajustado | 0.6531972647421809| 
| MAE ajustado | 0.3517948717948718 | 
| R² ajustado | 0.449534747610121   | 

- **Boa Sorte!** 

-->