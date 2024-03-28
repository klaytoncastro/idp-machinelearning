# Machine Learning - Desafio 01

Para enfatizar a importância da estatística descritiva em seu conjunto de dados, vamos abordar as características principais de cada variável. Isso inclui calcular medidas de tendência central (média, mediana), dispersão (desvio padrão, intervalo interquartil), além de explorar a distribuição de cada variável (contagem, valores únicos, possíveis outliers) e a relação entre as variáveis e a classe alvo (vinhos tintos/brancos e qualidade). 

- Utilize Pandas e Numpy para manipulação dos dados, Seaborn ou Matplotlib para visualização e Scikit-Learn para Machine Learning. 

- Carregue o Dataset em formato CSV, realize as operaçoes e manipulação dos dados conforme necessário em seus DataFrames. 

- Visualização Geral: Obtenha uma visão geral do dataset através do método `.describe()` para as variáveis numéricas e `.info()` para uma visão geral do tipo de dados e valores ausentes.

- Análise Descritiva: Obtenha as medidas de tendência central e dispersão para cada variável.

- Realize a contagem de valores para a variável categórica (color) e discreta (quality), alvos de nosso modelo de Machine Learning. 

- Realize a análise de distribuição de cada variável numérica usando histogramas.

- Realize a análise de Boxplots para identificar outliers.

- Conforme exposto em Sala de Aula, adotamos o algoritimo `ExtraTrees` para criar um modelo básico de Machine Learning para as tarefas de classificar os vinhos em tintos ou brancos e predizer a qualidade (nota) conforme análise de suas propriedades químicas. 

- Teste outros algoritmos para tarefas de classificação (color) e regressão (quality), conforme pesquisa do trabalho em grupo e a apresentação já realizada em sala de aula.    

- Otimize o desempenho utilizando como base as métricas obtidas no modelo original com `ExtraTreesClassifier` e, especialmente `ExtraTreesRegressor`, onde há maior margem para otimização. 

- Observe a relação com a Variável Alvo: Explore como as variáveis numéricas se relacionam com a variável alvo (quality). Quais variáveis podem ser removidas no modelo? Após remover essas variáveis, como se comportam as métricas de avaliação? 

- Os estudantes que conseguirem realizar uma otimização receberão pontuação extra! Boa Sorte! 