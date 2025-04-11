## 1. Introdução ao Uso de Datasets para Aprendizado de Máquina

Antes de aplicar algoritmos de aprendizado de máquina, é essencial entender o que são **dados**, como eles são organizados em **datasets**, e por que precisam ser cuidadosamente **preparados** para modelagem.

### 1.1. O que é um dataset?

Um dataset é uma coleção estruturada de dados, normalmente organizada em forma de tabela, onde:
- Cada linha representa um exemplo, instância ou registro;
- Cada coluna representa uma variável, característica ou atributo;
- Em tarefas supervisionadas, há uma coluna especial chamada **rótulo** ou **variável-alvo**, usada para treinamento do modelo.

### 1.2. A importância da preparação dos dados

Na prática, os dados raramente estão prontos para uso direto. Eles podem conter:
- **Valores ausentes**, **ruídos** ou **erros**
- **Tipos mistos de atributos** (numéricos, categóricos, binários)
- **Escalas incompatíveis** ou **valores discrepantes**
- **Redundâncias** ou **informações não relevantes**

Por isso, o **pré-processamento** é uma etapa crítica no pipeline de aprendizado de máquina. Esse processo envolve:
- **Limpeza de dados**: tratar valores ausentes, inconsistências e outliers
- **Transformação de dados**: normalizar escalas, converter variáveis categóricas, discretizar, codificar
- **Integração e enriquecimento**: combinar dados de fontes distintas e adicionar atributos úteis
- **Armazenamento e documentação**: salvar versões limpas e reprodutíveis para uso futuro

Essas etapas são abordadas nos conceitos de **Data Wrangling**, amplamente utilizados tanto em ciência de dados quanto em engenharia de machine learning.

---

## 2. Do Weka ao Python: por que atualizar a abordagem?

O **Weka** (Waikato Environment for Knowledge Analysis) é um software desenvolvido na Universidade de Waikato, Nova Zelândia, com o objetivo de tornar o aprendizado de máquina acessível por meio de uma interface gráfica. Entre os anos 2000 e 2010, foi amplamente utilizado no ensino por sua simplicidade e repositório de datasets clássicos.

No entanto, o ecossistema de **Python** com bibliotecas como `pandas`, `scikit-learn`, `matplotlib`, `seaborn` e `numpy` tornou-se o novo padrão de mercado por diversas razões:
- **Flexibilidade e automação** 
- **Reprodutibilidade científica**
- **Integração com projetos reais e pipelines robustos**
- **Amplo suporte da comunidade e evolução constante**

Neste repositório, alguns dos datasets clássicos originalmente utilizados no Weka foram convertidos para o formato `.csv` e reorganizados para uso com ferramentas mais modernas, como `pandas` e `scikit-learn`. Apesar da aparente simplicidade e do porte reduzido, esses conjuntos de dados ainda são extremamente úteis em contextos educacionais, pois permitem ilustrar com clareza os fundamentos do aprendizado de máquina:

- Introdução às tarefas supervisionadas de classificação e regressão;
- Exploração visual de padrões, outliers e distribuições de atributos;
- Prática de etapas de pré-processamento e construção de pipelines completos;
- Avaliação de modelos com diferentes métricas e estratégias de validação cruzada.

---

## 3. Datasets Clássicos

### 3.1. Iris Dataset — Classificação de Espécies de Flores

Número de instâncias: 150  
Número de atributos: 4 + classe  
Tipo de tarefa: Classificação multiclasse  
Fonte: https://archive.ics.uci.edu/ml/datasets/iris

Desafio:  
Prever a espécie da flor com base nas medidas da sépala e da pétala.

Atributos:
- sepal_length – Comprimento da sépala
- sepal_width – Largura da sépala
- petal_length – Comprimento da pétala
- petal_width – Largura da pétala
- class – Espécie da flor (setosa, versicolor, virginica)

Algoritmos sugeridos:
- k-NN, SVM, Decision Tree

Pipeline recomendado:
- Separar X e y
- Padronizar com StandardScaler
- Avaliar com accuracy_score, confusion_matrix
- Visualizar com PCA ou seaborn.pairplot

Objetivo:  
Explorar classificação supervisionada, visualização com PCA e overfitting em conjuntos bem separados.

---

### 3.2. Vote Dataset — Previsão de Partido Político com Base em Votação

Número de instâncias: 435  
Número de atributos: 16 + classe  
Tipo de tarefa: Classificação binária  
Fonte: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

Desafio:  
Prever o partido (democrata ou republicano) com base nos votos em 16 temas.

Atributos:
- 16 votos (atributos categóricos: y, n, ?)
- class – Partido do deputado (democrat, republican)

Algoritmos sugeridos:
- Naive Bayes, Logistic Regression, Random Forest

Pipeline recomendado:
- Tratar ? como np.nan, aplicar imputação
- Codificar variáveis categóricas
- Avaliar accuracy, precision, recall, f1

Objetivo:  
Trabalhar com dados categóricos, valores faltantes e análise de importância de variáveis políticas.

---

### 3.3. Diabetes Dataset — Detecção de Diabetes com Dados Clínicos

Número de instâncias: 768  
Número de atributos: 8 + classe  
Tipo de tarefa: Classificação binária  
Fonte: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Desafio:  
Prever se a paciente terá diabetes com base em exames clínicos.

Atributos:
- pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree_function, age
- class – 0 (não diabética) ou 1 (diabética)

Algoritmos sugeridos:
- Logistic Regression, Random Forest, XGBoost

Pipeline recomendado:
- Normalização
- Avaliar com confusion_matrix, classification_report, ROC-AUC

Objetivo:  
Analisar dados médicos com desbalanceamento, métricas médicas e importância de variáveis.

---

### 3.4. Ionosphere Dataset — Análise de Ecos de Radar

Número de instâncias: 351  
Número de atributos: 34 + classe  
Tipo de tarefa: Classificação binária  
Fonte: https://archive.ics.uci.edu/ml/datasets/ionosphere

Desafio:  
Classificar ecos de radar como bons ou ruins.

Atributos:
- 34 variáveis numéricas extraídas de sinais de radar
- class – g (bom) ou b (ruim)

Algoritmos sugeridos:
- SVM, k-NN, ExtraTreesClassifier

Pipeline recomendado:
- Padronização com MinMaxScaler ou RobustScaler
- Avaliação com cross_val_score

Objetivo:  
Praticar com dados inteiramente numéricos e alta dimensionalidade.

---

### 3.5. Segment Dataset — Classificação de Segmentos de Imagem

Número de instâncias: 2100  
Número de atributos: 19 + classe  
Tipo de tarefa: Classificação multiclasse  
Fonte: https://archive.ics.uci.edu/ml/datasets/image+segmentation

Desafio:  
Classificar segmentos de imagem com base em estatísticas de cor e textura.

Atributos:
- 19 atributos numéricos (valores médios, desvios, etc.)
- class – rótulo do segmento (brickface, sky, foliage...)

Algoritmos sugeridos:
- Random Forest, MLPClassifier, Gradient Boosting

Pipeline recomendado:
- Normalização
- Avaliação com classification_report
- Visualização com PCA

Objetivo:  
Aplicar modelos a um problema visual com dados tabulares derivados.

---

### 3.6. CPU with Vendor Dataset — Previsão de Performance de CPUs

Número de instâncias: 209  
Número de atributos: 8 + performance  
Tipo de tarefa: Regressão  
Fonte: https://archive.ics.uci.edu/ml/datasets/CPU+performance

Desafio:  
Prever tempo de execução do CPU com base nas suas características.

Atributos:
- vendor – Fabricante
- model – Modelo da CPU
- myct, mmin, mmax, cach, chmin, chmax – atributos de desempenho
- performance – variável alvo

Algoritmos sugeridos:
- Linear Regression, RandomForestRegressor, SVR

Pipeline recomendado:
- Encoding do vendor
- Avaliar com MAE, RMSE, R²
- Visualização: y_true vs y_pred

Objetivo:  
Introduzir regressão com variáveis mistas (categóricas e numéricas).

---

<!-- PROVA PRÁTICA - AV1

## 4. Prova Prática — 1ª Avaliação (AV1)

Nesta avaliação, você deverá demonstrar domínio sobre os principais tipos de tarefas em Aprendizado de Máquina: **classificação, regressão e clusterização**.

### 4.1. Instruções Gerais

Você deverá escolher e resolver **1 desafio de cada categoria**:

- **1 de classificação**
- **1 de regressão**
- **1 de clusterização**

Cada notebook (ou seção bem identificada) deve conter:

1. Descrição do problema  
2. Análise exploratória dos dados  
3. Estratégia de pré-processamento  
4. Escolha de modelo(s) e justificativa  
5. Avaliação dos resultados com métricas ou visualizações  
6. Conclusões interpretadas

> A entrega deve conter código funcional, bem comentado e com organização clara.

### 4.2. Datasets disponíveis

#### Classificação

| Dataset             | Descrição resumida                                     |
|----------------------|--------------------------------------------------------|
| `iris.csv`           | Previsão da espécie da flor pelas medidas              |
| `vote.csv`           | Previsão do partido com base nos votos em 16 temas     |
| `diabetes.csv`       | Previsão de diabetes com base em exames clínicos       |
| `ionosphere.csv`     | Classificação de ecos de radar                         |
| `segment-test.csv`   | Classificação de segmentos de imagem                   |
| `breast-cancer.csv`  | Diagnóstico de câncer (recorrente ou não)              |

#### Regressão

| Dataset                 | Descrição resumida                                       |
|--------------------------|----------------------------------------------------------|
| `cpu.with.vendor.csv`    | Previsão da performance de CPUs com variável categórica  |
| `machine.cpu.csv`        | Previsão da performance com atributos puramente numéricos|
| `auto-price.csv`         | Previsão do preço de veículos com base em atributos técnicos |

#### Clusterização

| Dataset             | Descrição resumida                                          |
|----------------------|-------------------------------------------------------------|
| `glass.csv`          | Agrupamento de tipos de vidro por composição química        |
| `segment-test.csv`   | Clusterização de segmentos com base em atributos visuais    |
| `soybean.csv`        | Agrupamento de características de doenças em plantas de soja|

---

### 4.3. Critérios de Avaliação

Cada desafio será avaliado de forma independente, considerando:

| Critério                                | Pontos por desafio |
|-----------------------------------------|--------------------|
| Organização e estrutura do notebook     | 1,0                |
| Qualidade da análise e pré-processamento| 1,0                |
| Aplicação e justificativa do modelo     | 1,0                |
| Avaliação dos resultados                | 1,0                |
| Clareza das conclusões                  | 1,0                |

**Total por desafio: 5,0 pontos × 3 = 15,0 pontos**  
A nota será proporcionalizada para o valor final da AV1.

---

### 4.4. Entrega

- **Prazo de entrega**: até **[DATA A DEFINIR PELO PROFESSOR]**
- Enviar os notebooks via e-mail ou repositório GitHub compartilhado

FIM DA PROVA -->