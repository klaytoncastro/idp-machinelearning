## 2. Datasets Convertidos do Weka

Este documento apresenta os **datasets educacionais convertidos do Weka para CSV**, com foco no uso em projetos de aprendizado de máquina com Python e scikit-learn. Cada dataset inclui:

- Descrição dos atributos
- Tarefa de ML associada
- Desafio prático
- Sugestões de algoritmos
- Etapas recomendadas de pipeline

---

### 2.1. iris.csv
**Tarefa**: Classificação multiclasse  
**Desafio**: Prever a espécie da flor com base em suas medidas.

**Atributos**:
- `sepal_length` – Comprimento da sépala
- `sepal_width` – Largura da sépala
- `petal_length` – Comprimento da pétala
- `petal_width` – Largura da pétala
- `class` – Espécie da flor (setosa, versicolor, virginica)

**Sugestão de algoritmos**:
- k-NN, SVM, Decision Tree

**Pipeline**:
- Separar X e y
- Padronizar (`StandardScaler`)
- Avaliar com `accuracy_score`, `confusion_matrix`
- Visualizar com `PCA` ou `seaborn.pairplot`

---

### 2.2. vote.csv
**Tarefa**: Classificação binária  
**Desafio**: Prever o partido (democrata ou republicano) com base nos votos em 16 temas.

**Atributos**:
- 16 votos (atributos categóricos: `y`, `n`, `?`)
- `class` – Partido do deputado (democrat, republican)

**Sugestão de algoritmos**:
- Naive Bayes, Logistic Regression, Random Forest

**Pipeline**:
- Tratar `?` como `np.nan`, aplicar imputação
- Codificar variáveis categóricas
- Avaliar `accuracy`, `precision`, `recall`, `f1`

---

### 2.3. diabetes.csv
**Tarefa**: Classificação binária  
**Desafio**: Prever se a paciente terá diabetes com base em exames clínicos.

**Atributos**:
- `pregnancies`, `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, `BMI`, `diabetes_pedigree_function`, `age`
- `class` – 0 (não diabética) ou 1 (diabética)

**Sugestão de algoritmos**:
- Logistic Regression, Random Forest, XGBoost

**Pipeline**:
- Normalização
- Avaliar com `confusion_matrix`, `classification_report`, `ROC-AUC`

---

### 2.4. ionosphere.csv
**Tarefa**: Classificação binária  
**Desafio**: Classificar ecos de radar como “bons” ou “ruins”.

**Atributos**:
- 34 variáveis numéricas extraídas de sinais de radar
- `class` – `g` (bom) ou `b` (ruim)

**Sugestão de algoritmos**:
- SVM, k-NN, ExtraTreesClassifier

**Pipeline**:
- Padronização com `MinMaxScaler` ou `RobustScaler`
- Avaliação com `cross_val_score`

---

### 2.5. segment-test.csv
**Tarefa**: Classificação multiclasse  
**Desafio**: Classificar segmentos de imagem com base em estatísticas de cor e textura.

**Atributos**:
- 19 atributos numéricos (valores médios, desvios, etc.)
- `class` – rótulo do segmento (brickface, sky, foliage...)

**Sugestão de algoritmos**:
- Random Forest, MLPClassifier, Gradient Boosting

**Pipeline**:
- Normalização
- Avaliação com `classification_report`, `PCA`

---

### 2.6. cpu.with.vendor.csv
**Tarefa**: Regressão  
**Desafio**: Prever tempo de execução do CPU com base nas suas características.

**Atributos**:
- `vendor` – Fabricante
- `model` – Modelo da CPU
- `myct`, `mmin`, `mmax`, `cach`, `chmin`, `chmax`
- `performance` – Variável alvo (tempo de execução)

**Sugestão de algoritmos**:
- Linear Regression, RandomForestRegressor, SVR

**Pipeline**:
- Encoding do `vendor`
- Avaliar com MAE, RMSE, R²
- Visualização: `y_true` vs `y_pre

<!--

## 4. Prova Prática — 1ª Avaliação (AV1)

Nesta primeira avaliação prática, você deverá demonstrar sua compreensão sobre tarefas supervisionadas e não supervisionadas em Machine Learning, utilizando os **datasets convertidos do Weka** já explorados em sala.

### 4.1. Instruções

- Escolha e resolva **2 (dois)** dos desafios listados abaixo (**obrigatórios**).
- A resolução de um **terceiro desafio** é **opcional** e poderá agregar até **2 pontos bônus** na nota final da AV1.
- Utilize `pandas`, `scikit-learn` e outras bibliotecas vistas em aula.
- Entregue um notebook `.ipynb` para cada desafio resolvido ou um único notebook bem organizado com todos os desafios.
- Capriche na explicação do processo, na justificativa das escolhas e na apresentação dos resultados.

### 4.2. Desafios disponíveis

| Dataset               | Tarefa                 | Descrição resumida                                         |
|------------------------|------------------------|-------------------------------------------------------------|
| `iris.csv`             | Classificação          | Prever a espécie da flor com base em medidas da sépala e pétala |
| `vote.csv`             | Classificação binária  | Prever o partido (democrata ou republicano) com base em 16 votos |
| `diabetes.csv`         | Classificação binária  | Prever se a paciente terá diabetes com base em exames clínicos |
| `ionosphere.csv`       | Classificação binária  | Classificar ecos de radar como bons ou ruins                  |
| `segment-test.csv`     | Classificação multiclasse | Classificar segmentos de imagem com base em estatísticas |
| `cpu.with.vendor.csv`  | Regressão              | Prever tempo de execução do CPU com base em seus atributos  |

### 4.3. Estrutura esperada do notebook

Cada notebook entregue deve conter:

1. Título do desafio e descrição do problema  
2. Análise exploratória inicial  
3. Estratégia de pré-processamento dos dados  
4. Escolha e aplicação de ao menos 1 algoritmo adequado  
5. Avaliação do modelo com métricas apropriadas  
6. Visualizações relevantes (quando aplicável)  
7. Conclusão com interpretações ou reflexões  

### 4.4. Critérios de Avaliação (por desafio)

| Critério                                 | Pontos |
|------------------------------------------|--------|
| Organização e estrutura do notebook      | 2,0    |
| Qualidade do pré-processamento           | 2,0    |
| Escolha e justificativa dos algoritmos   | 2,0    |
| Aplicação correta e avaliação do modelo  | 2,0    |
| Conclusões, insights e interpretação     | 2,0    |

**Total por 2 desafios obrigatórios: 10,0 pontos**  
**Desafio extra (opcional): até 2,0 pontos bônus**

### 4.5. Entrega

- Prazo: até **[DATA LIMITE]** (a ser definido pelo professor)
- Enviar por e-mail ou via repositório GitHub (link compartilhado)

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

-->