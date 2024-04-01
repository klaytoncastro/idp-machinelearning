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

- Teste outros algoritmos para tarefas de classificação (color) e regressão (quality), conforme a pesquisa em grupo apresentada em sala de aula. 

- Observe a relação com a variável alvo: explore como as variáveis numéricas se relacionam com a variável alvo (quality). Quais variáveis podem ser removidas no modelo? Após remover essas variáveis, como se ajusta o modelo e se comportam as métricas de avaliação? 

- Otimize o desempenho utilizando como base as métricas obtidas no modelo original com `ExtraTreesClassifier` e, especialmente `ExtraTreesRegressor`, onde há maior margem para otimização. 

- Os estudantes que conseguirem realizar uma otimização receberão pontuação extra! 

| Métrica | Valor                 |
|---------|-----------------------|
| Acurácia| 0.54281651011084      |
| MSE     | 0.3498934358974359    |
| RMSE    | 0.5915179083488816    |
| MAE     | 0.39295384615384615   |
| R2      | 0.54281651011084      |

- **Boa Sorte!** 