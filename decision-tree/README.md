# Machine Learning - Desafio 01

É importante abordar a estatística descritiva para enfatizar as características de cada variável (impacto das features) em seu conjunto de dados (dataset). Isso inclui calcular medidas de tendência central (média, mediana), dispersão (desvio padrão, intervalo interquartil), além de explorar a distribuição de cada variável (contagem, valores únicos, possíveis outliers), bem como a relação entre as variáveis preditoras e a variável alvo. 

- Utilize Pandas e Numpy para manipulação dos dados, Seaborn ou Matplotlib para visualização e Scikit-Learn para criar seu modelo de Machine Learning. 

- Realize a análise de distribuição de cada variável numérica usando histogramas.

- Realize a análise de Boxplots para identificar outliers.

- Carregue o Dataset em formato CSV, realize as operaçoes e manipulação dos dados conforme necessário em seus DataFrames. 

- Visualização Geral: Obtenha uma visão geral do dataset através dos métodos `.describe()` e `.info()` para uma visão geral do tipo de dados e valores ausentes.

- Análise Descritiva: Obtenha as medidas de tendência central e dispersão para cada variável.

- Realize a contagem de valores para a variável categórica (color) e discreta (quality), alvos de nosso modelo de Machine Learning. 

- Nosso primeiro objetivo era prever a cor do vinho (branco ou tinto), uma tarefa de classificação. Em seguida, prever a qualidade do vinho, uma tarefa de regressão. 

- Para isso adotamos o algoritimo `ExtraTrees` para criar um modelo básico de Machine Learning, utilizando o conceito de árvore de decisão para as tarefas de classificar os vinhos em tintos ou brancos e, em seguida, para predizer a qualidade (nota) conforme análise de suas propriedades químicas. 

## Tarefa 01: 

- Teste outros algoritmos para tarefas de classificação (color) e regressão (quality), conforme a pesquisa em grupo apresentada em sala de aula. 

- Para problemas de classificação, além do algoritmo `ExtraTreesClassifier`, que faz uma robusta implementação baseada em Árvore de Decisão, `Naive Bayes` e `Support Vector Machine (SVM)` são alternativas populares, dependendo da natureza dos dados e do problema específico que você está tentando resolver. 

### Usando Naive Bayes para Classificação

O Naive Bayes é uma técnica de classificação baseada em aplicar o teorema de Bayes com a "ingenuidade" de supor independência entre os preditores. É fácil de construir e particularmente útil para grandes volumes de dados. Além disso, é eficaz em problemas de classificação multinomial e binomial. 

Existem diferentes implementações de Naive Bayes no Scikit-Learn, adequados para diferentes tipos de dados:

- GaussianNB: Usado em classificação onde as features são contínuas e seguem uma distribuição normal.
- MultinomialNB: Bom para quando suas features são contagens ou frequências de termos (comumente usado em classificação de texto).
- BernoulliNB: Adequado para features binárias.

Teste as implementações e avalie os resultados: 

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

### Avaliação dos Modelos

Após treinar o seu modelo, você precisa avaliar o quão bem ele performa. Para classificação, vimos que métricas comuns incluem acurácia, precisão, recall, e a F1-score. O Scikit-Learn fornece funções prontas para calcular essas métricas:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
```

## Tarefa 02

Para a tarefa de regressão — prever um número inteiro, no caso a qualidade do vinho, existem vários algoritmos no Scikit-Learn além da implementação robusta de Árvore de Decisão com o `ExtraTreesRegressor`. Dentre as alternativas populares temos: 

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

### Avaliação

Ao invés de Precision, Recall, F1-Score, que são métricas adequadas para tarefas de classificação, utilize MSE, RMSE, MAE e coeficiente R2, que são adequadas a um problema de regressão. 

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

## Tarefa 03

- Observe a relação com a variável alvo: explore como as variáveis se relacionam. Quais variáveis podem ser removidas no modelo para previsão da qualidade? 

- Após remover essas variáveis (utilize o Pandas para isso), como se ajusta o modelo e se comportam as métricas de avaliação? 

- Otimize o desempenho utilizando como base as métricas obtidas na tarefa de regressão, onde há maior margem para otimização. 

- Os estudantes que conseguirem realizar uma otimização receberão pontuação extra! 

| **Métrica** | **Valor**          |
|---------|------------------------|
| Acurácia| 0.54281651011084       |
| MSE     | 0.3498934358974359     |
| RMSE    | 0.5915179083488816     |
| MAE     | 0.39295384615384615    |
| R2      | 0.54281651011084       |
| **Pós-Arrendondamento**          |
| Acurácia ajustada | 0.9702564102564103
| MSE ajustado | 0.4266666666666667 | 
| RMSE ajustado | 0.6531972647421809| 
| MAE ajustado | 0.3517948717948718 | 
| R² ajustado | 0.449534747610121   | 

- **Boa Sorte!** 