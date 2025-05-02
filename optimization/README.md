# 1. Introdução ao Uso de Datasets para Aprendizado de Máquina

Antes de aplicar algoritmos de aprendizado de máquina, é essencial entender o que são **dados**, como eles são organizados em **datasets**, e por que precisam ser cuidadosamente **preparados** para modelagem.

## 1.1. O que é um dataset?

Um dataset é uma coleção estruturada de dados, normalmente organizada em forma de tabela, onde:
- Cada linha representa um exemplo, instância ou registro;
- Cada coluna representa uma variável, característica ou atributo;
- Em tarefas supervisionadas, há uma coluna especial chamada **rótulo** ou **variável-alvo**, usada para treinamento do modelo.

## 1.2. A importância da preparação dos dados

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

# 2. Do Weka ao Python: por que atualizar a abordagem?

O **Weka** é uma ferramenta didática clássica, mas hoje o ecossistema Python com bibliotecas como `pandas`, `scikit-learn`, `matplotlib`, `seaborn` e `numpy` é o padrão de mercado por:

- **Flexibilidade e automação**
- **Reprodutibilidade científica**
- **Integração com projetos reais e pipelines robustos**
- **Amplo suporte da comunidade e evolução constante**

---

# 3. Datasets Clássicos Utilizados

## 3.1. Iris Dataset
- Tipo: Classificação multiclasse  
- Instâncias: 150  
- Atributos: 4 + classe  
- Fonte: UCI

## 3.2. Vote Dataset
- Tipo: Classificação binária  
- Instâncias: 435  
- Atributos: 16 + classe  
- Fonte: UCI

## 3.3. Diabetes Dataset
- Tipo: Classificação binária  
- Instâncias: 768  
- Atributos: 8 + classe  
- Fonte: Kaggle

## 3.4. Ionosphere Dataset
- Tipo: Classificação binária  
- Instâncias: 351  
- Atributos: 34 + classe  
- Fonte: UCI

## 3.5. Segment Dataset
- Tipo: Classificação ou Clusterização  
- Instâncias: 2100  
- Atributos: 19 + classe  
- Fonte: UCI

## 3.6. CPU with Vendor Dataset
- Tipo: Regressão  
- Instâncias: 209  
- Atributos: 8 + performance  
- Fonte: UCI

## 3.7. Telecom Dataset
- Tipo: Clusterização  
- Fonte: Convertido de ARFF

## 3.8. AirQuality Dataset
- Tipo: Regressão  
- Fonte: UCI

## 3.9. Bank Dataset
- Tipo: Classificação e Clusterização  
- Fonte: WekaLearningDataset

---

# 4. Prova Prática — 1ª Avaliação (AV1)

## 4.1. Instruções

Escolha e resolva **1 desafio de cada tipo**:

- 1 de Classificação
- 1 de Regressão
- 1 de Clusterização

Cada notebook deve conter:
1. Descrição do problema  
2. Análise exploratória dos dados  
3. Estratégia de pré-processamento  
4. Escolha de modelo(s) e justificativa  
5. Avaliação dos resultados com métricas ou visualizações  
6. Conclusões interpretadas

## 4.2. Datasets disponíveis

### Classificação 
- `iris.csv`  
- `vote.csv`  
- `diabetes.csv`  
- `ionosphere.csv`  
- `segment-test.csv`  
- `bank.csv`  

### Regressão
- `cpu.with.vendor.csv`  
- `airquality.csv`  

### Clusterização
- `segment-test.csv`  
- `telecom.csv`  
- `bank.csv`  

## 4.3. Avaliação

| Critério                                | Pontos por desafio |
|-----------------------------------------|--------------------|
| Organização e estrutura do notebook     | 1.0                |
| Qualidade da análise e pré-processamento| 1.0                |
| Aplicação e justificativa do modelo     | 1.0                |
| Avaliação dos resultados                | 1.0                |
| Clareza das conclusões                  | 1.0                |

Total por desafio: **5.0 pontos × 3 = 15.0 pontos**  
Nota será proporcionalizada.

## 4.4. Bônus (até +1.0 ponto)

Implemente **otimização de hiperparâmetros** com:

- `GridSearchCV`  
- `RandomizedSearchCV`  
- `Otimização Bayesiana` 

> Apresente e compare os resultados. Mostre os melhores parâmetros e o impacto nas métricas.

---

# 5. Otimização de Hiperparâmetros em Machine Learning

## 5.1. Métodos

- **Grid Search**  
- **Randomized Search**  
- **Otimização Bayesiana (com skopt)**

## 5.2. Algoritmos Sugeridos

- Decision Tree  
- Random Forest  
- Extra Tree  
- Extra Trees  
- XGBoost  
- LightGBM  
- Naive Bayes  
- Logistic Regression  
- SVM  
- k-NN  

## 5.3. Métricas

- **Acurácia**  
- **Precisão**  
- **Recall**  
- **F1-Score**

Para regressão:
- **MAE**
- **MSE**
- **RMSE**
- **R²**

## 5.4. Relatório Sucinto no Notebook (Markdown)

1. Introdução ao Algoritmo  
2. Metodologia  
3. Resultados com gráficos e comparação  
4. Discussão sobre impacto e custo computacional

---

**Boa sorte!**
