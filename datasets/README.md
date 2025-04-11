# Datasets Convertidos do Weka

Este documento apresenta os **datasets educacionais convertidos do Weka para CSV**, com foco no uso em projetos de aprendizado de máquina com Python e scikit-learn. Cada dataset inclui:

- Descrição dos atributos
- Tarefa de ML associada
- Desafio prático
- Sugestões de algoritmos
- Etapas recomendadas de pipeline

---

## 1. iris.csv
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

## 2. vote.csv
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

## 3. diabetes.csv
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

## 4. ionosphere.csv
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

## 5. segment-test.csv
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

## 6. cpu.with.vendor.csv
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
- Visualização: `y_true` vs `y_pred`
