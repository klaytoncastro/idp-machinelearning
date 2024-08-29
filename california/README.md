# Elaborando uma Abordagem de Machine Learning para uma Tarefa de Regressão

## Descrição do Dataset

O dataset **California Housing** é um conjunto de dados clássico para tarefas de regressão, utilizado para prever o valor mediano das casas em várias regiões da Califórnia. Você pode importar o dataset diretamente do seguinte link:

[Dataset California Housing](https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/california/california_housing.csv)

### Atributos do Dataset:

- **MedInc**: Renda mediana dos residentes do bairro (em dezenas de milhares de dólares).
- **HouseAge**: Idade mediana das casas no bairro.
- **AveRooms**: Número médio de cômodos por casa.
- **AveBedrms**: Número médio de quartos por casa.
- **Population**: População do bairro.
- **AveOccup**: Número médio de ocupantes por casa.
- **Latitude**: Latitude do bairro.
- **Longitude**: Longitude do bairro.

### Variável Alvo:

- **MedHouseVal**: Valor mediano das casas no bairro (em centenas de milhares de dólares).

## Tarefa:  

Utilize o notebook apresentado em sala de aula [Air Quality](https://github.com/klaytoncastro/idp-machinelearning/blob/main/airquality/regressao_airquality_v2.ipynb) como referência para o fluxo de trabalho, considerando as etapas a seguir: 

### Análise Exploratória de Dados (EDA)

- **Distribuição dos Dados**: Analise a distribuição de cada variável para identificar outliers e entender as características gerais do conjunto de dados.
- **Correlação**: Explore as correlações entre as variáveis preditoras e a variável alvo, bem como as correlações entre as variáveis preditoras.

### Pré-processamento de Dados

1. **Tratamento de Valores Ausentes**: Verifique se há valores ausentes e trate-os adequadamente (caso haja).
2. **Normalização/Padronização**: Se necessário, normalize ou padronize os dados numéricos para garantir que todas as variáveis estejam na mesma escala.
3. **Divisão de Dados**: Separe o dataset em conjuntos de treino e teste para validação dos modelos.

### Modelagem

Aplique diferentes algoritmos de machine learning para tarefas de regressão, como:

- **Regressão Linear**: Como ponto de partida, utilize a regressão linear.
- **Árvores de Decisão**: Para capturar interações complexas entre as variáveis.
- **Random Forest**: Para melhorar a performance e reduzir o overfitting.
- **Support Vector Regressor (SVR)**: O SVR é uma extensão do SVM (Support Vector Machine) para tarefas de regressão.
- **Gradient Boosting Machines (GBM)**: Constrói modelos sequencialmente para corrigir erros dos modelos anteriores. **XGBoost** e **LightGBM** são implementações populares.

### Interpretação e Avaliação do Modelo

- **Interpretação dos Coeficientes**: Para modelos lineares, interprete os coeficientes para entender a relação entre as variáveis preditoras e o valor da casa.
- **Métricas de Avaliação**: Avalie o desempenho dos modelos usando métricas como o erro médio absoluto (MAE), o erro quadrático médio (MSE) e o coeficiente de determinação (R²).
- **Análise de Erros**: Explore os resíduos do modelo para entender onde e por que o modelo pode estar errando.
- **Gráfico de Valores Previstos x Valores Atuais**: Para cada modelo de regressão que você implementou, gere um gráfico de dispersão (scatter plot) comparando os valores previstos com os valores atuais. Isso ajudará a visualizar o desempenho do modelo.
