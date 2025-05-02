
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

## 3.1. Iris Dataset — Classificação de Espécies de Flores
- Tipo: Classificação multiclasse  
- Instâncias: 150  
- Atributos: 4 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/iris

Atributos:
- sepal_length – Comprimento da sépala (cm)
- sepal_width – Largura da sépala (cm)
- petal_length – Comprimento da pétala (cm)
- petal_width – Largura da pétala (cm)
- class – Espécie da flor (setosa, versicolor, virginica)

Desafio: prever a espécie da flor com base nas medidas morfológicas.

---

## 3.2. Vote Dataset — Previsão de Partido Político
- Tipo: Classificação binária  
- Instâncias: 435  
- Atributos: 16 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

Atributos:
- 16 variáveis indicando votos (y, n, ?)
- class – Partido (democrat, republican)

Desafio: prever o partido político de um congressista com base em seus votos.

---

## 3.3. Diabetes Dataset — Risco de Diabetes
- Tipo: Classificação binária  
- Instâncias: 768  
- Atributos: 8 + classe  
- Fonte: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Atributos:
- pregnancies – Número de gestações
- glucose – Nível de glicose
- blood_pressure – Pressão arterial
- skin_thickness – Espessura da pele
- insulin – Nível de insulina
- BMI – Índice de massa corporal
- diabetes_pedigree_function – Histórico familiar
- age – Idade
- class – 1 (diabética), 0 (não)

Desafio: prever se uma paciente tem diabetes com base em dados clínicos.

---

## 3.4. Ionosphere Dataset — Sinais de Radar
- Tipo: Classificação binária  
- Instâncias: 351  
- Atributos: 34 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/ionosphere

Atributos:
- 34 atributos numéricos derivados de sinais de radar
- class – g (bom), b (ruim)

Desafio: prever se um eco de radar representa uma estrutura válida.

---

## 3.5. Segment Dataset — Segmentos de Imagem
- Tipo: Classificação ou Clusterização  
- Instâncias: 2100  
- Atributos: 19 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/image+segmentation

Atributos: estatísticas de cor, simetria e textura extraídas de imagens.

Desafio: classificar ou agrupar segmentos de imagem com base em seus atributos visuais.

---

## 3.6. CPU with Vendor Dataset — Performance de CPUs
- Tipo: Regressão  
- Instâncias: 209  
- Atributos: 8 + performance  
- Fonte: https://archive.ics.uci.edu/ml/datasets/CPU+performance

Atributos:
- vendor – Fabricante
- model – Modelo
- myct – Tempo de ciclo (ns)
- mmin – Memória mínima (KB)
- mmax – Memória máxima (KB)
- cach – Tamanho do cache (KB)
- chmin – Canais mínimos
- chmax – Canais máximos
- performance – Tempo de execução da CPU

Desafio: prever a performance da CPU com base em suas características técnicas.

---

## 3.7. Telecom Dataset — Clusterização de Clientes
- Tipo: Clusterização  
- Fonte: Convertido de ARFF

Atributos: características de consumo como chamadas internacionais, número de minutos, plano, ligações ao suporte etc.

Desafio: agrupar clientes com base em seu perfil de consumo e serviços contratados.

---

## 3.8. AirQuality Dataset — Poluição do Ar
- Tipo: Regressão  
- Fonte: https://archive.ics.uci.edu/ml/datasets/Air+Quality

Atributos:
- CO(GT), NMHC(GT), C6H6(GT), NOx(GT), NO2(GT): poluentes
- PT08.S1 a PT08.S5: leituras dos sensores
- T, RH, AH: temperatura, umidade relativa e absoluta
- Date, Time

Desafio: prever a concentração de poluentes com base nas medições ambientais.

---

## 3.9. Bank Dataset — Perfil de Clientes
- Tipo: Classificação e Clusterização  
- Fonte: https://github.com/bluenex/WekaLearningDataset/blob/master/bank/bank-data.csv

Atributos:
- age, sex, region, income, married, children, car, save_act, current_act, mortgage, pep

Desafio: prever se o cliente aderirá a um produto financeiro (PEP) ou segmentá-lo por perfil.

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
4. Escolha do(s) algoritmo(s) para o modelo e justificativa  
5. Avaliação dos resultados com métricas e visualizações  
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
- `Bayesian Optimization` 

> Apresente e compare os resultados. Mostre os melhores parâmetros e o impacto nas métricas.

---

# 5. Escolha de Algoritmos e Otimização de Hiperparâmetros

## 5.1. Algoritmos Sugeridos

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

## 5.2. Métodos

- **Grid Search**  
- **Randomized Search**  
- **Otimização Bayesiana**

## 5.3. Métricas

Para classificação: 
- **Acurácia**  
- **Precisão**  
- **Recall**  
- **F1-Score**

Para regressão:
- **MAE**
- **MSE**
- **RMSE**
- **R²**

Para clusterização:
- **k-Elbow**
- **Silhouette Score**

---

**Boa sorte!**
