# Fluxo de Trabalho e Otimização em Aprendizado de Máquina

## 1. Objetivo da Tarefa

Nesta tarefa, cada aluno deverá desenvolver três fluxos completos de aprendizado de máquina, contemplando três tipos fundamentais de problema:

- **Classificação**
- **Regressão**
- **Clusterização**

O foco é entender o processo, justificar as escolhas e interpretar os resultados, praticando o ciclo completo de trabalho em machine learning:

```text
→ entendimento do problema
→ escolha do dataset
→ análise exploratória
→ pré-processamento
→ modelagem inicial
→ avaliação
→ otimização de hiperparâmetros
→ comparação dos resultados
→ conclusão
```

## 2. Entrega

A entrega deverá ser feita no Canvas, em **3 notebooks distintos**, no formato `.ipynb`, contemplando obrigatoriamente:

| Notebook | Tipo de problema | Objetivo |
|---|---|---|
| 1 | Classificação | Prever uma classe ou categoria |
| 2 | Regressão | Prever um valor numérico contínuo |
| 3 | Clusterização | Identificar grupos ou padrões nos dados |

Nomenclatura sugerida:

```text
classificacao_nome_sobrenome.ipynb
regressao_nome_sobrenome.ipynb
clusterizacao_nome_sobrenome.ipynb
```

## 3. Datasets disponíveis

```text
https://github.com/klaytoncastro/idp-machinelearning/tree/main/datasets
```

Exemplos com vários algoritmos e implementação das técnicas de otimização de hiperparâmetros fundamentais (grid search, randomized search, bayesian search):

```text
https://github.com/klaytoncastro/idp-machinelearning/tree/main/optimization
```

O aluno poderá utilizar os datasets disponíveis no repositório da disciplina ou escolher outros datasets. A entrega deve conter análise própria, justificativas e interpretação dos resultados.

### 3.1. Datasets para Classificação

| Dataset | Arquivo sugerido | Descrição | Desafio |
|---|---|---|---|
| Iris | `iris.csv` | Medidas de flores de três espécies diferentes | Prever a espécie da flor |
| Vote | `vote.csv` | Votos de congressistas em pautas legislativas | Prever o partido político |
| Diabetes | `diabetes.csv` | Dados clínicos de pacientes | Prever ocorrência de diabetes |
| Ionosphere | `ionosphere.csv` | Sinais de radar classificados como bons ou ruins | Classificar o sinal de radar |
| Segment | `segment-test.csv` | Atributos extraídos de segmentos de imagem | Classificar o tipo de segmento |
| Bank | `bank-data.csv` | Perfil socioeconômico de clientes bancários | Prever adesão a produto financeiro |

### 3.2. Datasets para Regressão

| Dataset | Arquivo sugerido | Descrição | Desafio |
|---|---|---|---|
| CPU with Vendor | `cpu.with.vendor.csv` | Características técnicas de CPUs | Prever desempenho da CPU |
| AirQuality | `AirQualityUCI.csv` | Leituras ambientais e sensores químicos | Prever concentração de poluentes ou variável ambiental |
| Ames Housing | [Kaggle](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset) | Dados de imóveis com aproximadamente 79 atributos | Prever preço de venda |
| Energy Efficiency | [UCI](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) | Dados de eficiência energética de edifícios | Prever carga de aquecimento ou resfriamento |

### 3.3. Datasets para Clusterização

| Dataset | Arquivo sugerido | Descrição | Desafio |
|---|---|---|---|
| Segment | `segment-test.csv` | Atributos visuais de segmentos de imagem | Agrupar segmentos semelhantes |
| Telecom | `telecom-customer-data.csv` | Perfil e consumo de clientes de telecomunicações | Agrupar clientes por perfil de consumo |
| Bank | `bank-data.csv` | Perfil socioeconômico e bancário de clientes | Segmentar clientes por perfil |

## 4. Fluxo Obrigatório de Cada Notebook

Cada notebook deverá seguir o fluxo abaixo.

### 4.1. Definição do Problema

O aluno deverá explicar:

- qual problema será resolvido;
- qual dataset foi escolhido;
- qual é o tipo de problema;
- qual é a variável-alvo, quando houver;
- qual é o objetivo do modelo.

Exemplos:

- Classificação: prever a espécie de uma flor, o partido político ou a presença de diabetes.
- Regressão: prever desempenho de CPU ou concentração de poluentes.
- Clusterização: agrupar clientes ou segmentos de imagem por similaridade.

### 4.2. Carregamento do Dataset

O aluno deverá carregar o dataset escolhido e apresentar informações iniciais, como:

- quantidade de linhas e colunas;
- nomes das colunas;
- tipos de dados;
- primeiras linhas do dataset;
- descrição inicial das variáveis.


### 4.3. Análise Exploratória dos Dados

O aluno deverá realizar uma análise exploratória, incluindo, conforme aplicável:

- estatísticas descritivas;
- verificação de valores ausentes;
- distribuição das variáveis;
- análise de variáveis categóricas;
- análise de correlação;
- identificação de possíveis outliers;
- análise da variável-alvo, quando houver;
- gráficos ou visualizações relevantes.

O objetivo desta etapa é entender os dados antes de aplicar algoritmos e definir os modelos.

### 4.4. Pré-processamento

O aluno deverá preparar os dados para modelagem, aplicando as etapas necessárias, como:

- tratamento de valores ausentes;
- conversão de variáveis categóricas;
- normalização ou padronização;
- remoção de atributos irrelevantes;
- separação entre atributos preditores e variável-alvo;
- divisão entre treino e teste, quando for problema supervisionado.

Para clusterização, o aluno deverá justificar quais atributos serão utilizados para formar os grupos.

### 4.5. Modelagem Inicial

O aluno deverá aplicar pelo menos um modelo adequado ao tipo de problema e justificar sua escolha.

#### Classificação

Exemplos de algoritmos:

- Decision Tree;
- Random Forest;
- Logistic Regression;
- Naive Bayes;
- SVM;
- k-NN;
- Extra Trees;
- XGBoost;
- LightGBM.

#### Regressão

Exemplos de algoritmos:

- Linear Regression;
- Decision Tree Regressor;
- Random Forest Regressor;
- Extra Trees Regressor;
- Gradient Boosting Regressor;
- SVR;
- k-NN Regressor;
- XGBoost Regressor;
- LightGBM Regressor.

#### Clusterização

Exemplos de algoritmos:

- K-Means;
- DBSCAN;
- Agglomerative Clustering;
- Gaussian Mixture Models.

## 5. Avaliação dos Resultados

Cada tipo de problema exige formas diferentes de avaliação.

### 5.1. Avaliação em Classificação

O aluno deverá avaliar o modelo usando métricas como:

- acurácia;
- precisão;
- recall;
- F1-score;
- matriz de confusão.

Também deverá interpretar os resultados, explicando quais classes foram melhor ou pior classificadas.

### 5.2. Avaliação em Regressão

O aluno deverá avaliar o modelo usando métricas como:

- MAE;
- MSE;
- RMSE;
- R².

Também deverá interpretar o erro obtido e discutir se o modelo apresenta bom desempenho para o problema escolhido.

### 5.3. Avaliação em Clusterização

O aluno deverá avaliar os agrupamentos usando recursos como:

- método do cotovelo, como heurística para escolha do número de clusters;
- Silhouette Score;
- Davies-Bouldin Index;
- visualização dos clusters;
- interpretação dos grupos encontrados.

O aluno deverá explicar o que os clusters representam e se os agrupamentos fazem sentido em relação aos atributos utilizados.

## 6. Otimização de Hiperparâmetros

Além da modelagem inicial, o aluno deverá aplicar técnicas de otimização de hiperparâmetros nos problemas supervisionados, isto é, classificação e regressão.

As três abordagens abaixo deverão ser demonstradas nos notebooks de Classificação e/ou Regressão:

- `GridSearchCV`;
- `RandomizedSearchCV`;
- Otimização Bayesiana.

Estes métodos, contudo, não são geralmente adequados aos problemas não supervisionados, como na clusterização. No entanto, você deve explorar ajustes de parâmetros como:

- k-Elbow para definir o número de clusters no K-Means ou K-Medoids;
- inicialização dos centróides;
- parâmetros `eps` e `min_samples` no DBSCAN;

Nesses casos, a avaliação poderá se valer de métricas como `Silhouette Score`, `Davies-Bouldin Index` ou análise visual dos agrupamentos para justificar a escolha dos parâmetros.

### 6.1. Grid Search

O aluno deverá aplicar `GridSearchCV` em pelo menos um modelo.

Deve apresentar:

- o modelo escolhido;
- a grade de hiperparâmetros testada;
- os melhores parâmetros encontrados;
- a métrica obtida;
- comparação com o modelo inicial.

### 6.2. Randomized Search

O aluno deverá aplicar `RandomizedSearchCV` em pelo menos um modelo.

Deve apresentar:

- o modelo escolhido;
- o espaço de busca utilizado;
- número de combinações testadas;
- melhores parâmetros encontrados;
- comparação com o modelo inicial e com o Grid Search.

### 6.3. Otimização Bayesiana

O aluno deverá aplicar uma abordagem de otimização bayesiana, por exemplo com `skopt`.

Deve apresentar:

- o modelo escolhido;
- o espaço de busca;
- os melhores parâmetros encontrados;
- a métrica obtida;
- comparação com Grid Search e Randomized Search;
- discussão sobre desempenho e custo computacional.

## 7. Comparação Final e Documentação da Decisão

O custo computacional poderá ser estimado de forma simples, usando recursos como `%time`, `%timeit` ou a biblioteca `time`, registrando o tempo aproximado de execução das buscas de hiperparâmetros. Ao final, seu relatório deve apresentar uma comparação entre:

- modelo inicial;
- modelo otimizado com Grid Search;
- modelo otimizado com Randomized Search;
- modelo otimizado com Otimização Bayesiana.

Essa comparação deve mostrar:

- melhores hiperparâmetros encontrados;
- métricas antes e depois da otimização;
- impacto no desempenho;
- custo computacional aproximado;
- qual abordagem produziu o melhor resultado.

A conclusão deve responder:

- o problema foi resolvido de forma satisfatória?
- quais modelos tiveram melhor desempenho?
- a otimização melhorou os resultados?
- quais limitações foram observadas?
- o que poderia ser melhorado em uma próxima versão?

## 8. Avaliação

A tarefa será avaliada considerando a qualidade geral do fluxo desenvolvido nos três notebooks. Serão observados:

- organização dos notebooks;
- clareza na definição dos problemas;
- qualidade da análise exploratória;
- coerência do pré-processamento;
- escolha adequada dos modelos;
- correta aplicação das métricas;
- aplicação das técnicas de otimização;
- comparação entre os resultados;
- interpretação técnica das conclusões;
- reprodutibilidade dos notebooks.

A nota será atribuída de forma global, considerando a consistência da solução nos três tipos de problema. A tarefa não consiste apenas em executar algoritmos prontos. O aluno deve demonstrar domínio do fluxo completo de aprendizado de máquina:

```text
→ problema
→ dataset
→ análise exploratória
→ pré-processamento
→ modelo inicial
→ avaliação
→ otimização
→ comparação
→ conclusão
```
