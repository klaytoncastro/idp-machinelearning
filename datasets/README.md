
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

O **Weka** é uma ferramenta didática clássica, mas hoje o ecossistema Python com bibliotecas como `pandas`, `scikit-learn`, `matplotlib`, `seaborn` e `numpy` é o padrão de mercado por:

- **Flexibilidade e automação**
- **Reprodutibilidade científica**
- **Integração com projetos reais e pipelines robustos**
- **Amplo suporte da comunidade e evolução constante**

---

## 3. Datasets Clássicos Utilizados

### 3.1. Iris Dataset — Classificação de Espécies de Flores

- Tipo: Classificação multiclasse  
- Instâncias: 150  
- Atributos: 4 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/iris

Desafio: prever a espécie da flor a partir das medidas de sépala e pétala.

---

### 3.2. Vote Dataset — Previsão de Partido Político

- Tipo: Classificação binária  
- Instâncias: 435  
- Atributos: 16 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

Desafio: prever o partido (democrata ou republicano) com base em votos parlamentares.

---

### 3.3. Diabetes Dataset — Risco de Diabetes

- Tipo: Classificação binária  
- Instâncias: 768  
- Atributos: 8 + classe  
- Fonte: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Desafio: prever risco de diabetes com base em dados clínicos.

---

### 3.4. Ionosphere Dataset — Sinais de Radar

- Tipo: Classificação binária  
- Instâncias: 351  
- Atributos: 34 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/ionosphere

Desafio: classificar ecos de radar como bons ou ruins.

---

### 3.5. Segment Dataset — Segmentos de Imagem

- Tipo: Classificação ou Clusterização  
- Instâncias: 2100  
- Atributos: 19 + classe  
- Fonte: https://archive.ics.uci.edu/ml/datasets/image+segmentation

Desafio: classificar ou agrupar segmentos visuais com base em atributos tabulares.

---

### 3.6. CPU with Vendor Dataset — Performance de CPUs

- Tipo: Regressão  
- Instâncias: 209  
- Atributos: 8 + performance  
- Fonte: https://archive.ics.uci.edu/ml/datasets/CPU+performance

Desafio: prever tempo de execução de CPU com base em atributos técnicos e fabricante.

---

### 3.7. Telecom Dataset — Segmentação de Clientes

- Tipo: Clusterização  
- Instâncias: (a ser conferido no arquivo)  
- Atributos: (a confirmar no cabeçalho)  
- Fonte: convertido de `.arff` para `.csv`

Desafio: agrupar clientes por padrões de consumo e perfil (plano internacional, minutos, chamadas ao suporte etc.)

---

<!-- PROVA PRÁTICA -->

## 4. Prova Prática — 1ª Avaliação (AV1)

Nesta avaliação, você deverá demonstrar domínio sobre os principais tipos de tarefas em Aprendizado de Máquina: **classificação, regressão e clusterização**.

### 4.1. Instruções

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

> A entrega deve conter código funcional, bem comentado e com organização clara.

---

### 4.2. Datasets disponíveis

#### Classificação
- `iris.csv` – Espécie da flor  
- `vote.csv` – Partido político  
- `diabetes.csv` – Risco de diabetes  
- `ionosphere.csv` – Sinais de radar  
- `segment-test.csv` – Segmento de imagem

#### Regressão
- `cpu.with.vendor.csv` – Performance de CPU

#### Clusterização
- `segment-test.csv` – Atributos visuais  
- `telecom.csv` – Segmentação de clientes

---

### 4.3. Avaliação

| Critério                                | Pontos por desafio |
|-----------------------------------------|--------------------|
| Organização e estrutura do notebook     | 1.0                |
| Qualidade da análise e pré-processamento| 1.0                |
| Aplicação e justificativa do modelo     | 1.0                |
| Avaliação dos resultados                | 1.0                |
| Clareza das conclusões                  | 1.0                |

Total por desafio: **5.0 pontos × 3 = 15.0 pontos**  
Nota será proporcionalizada.

---

### 4.4. Bônus (até +1.0 ponto)

Implemente **otimização de hiperparâmetros** com:

- `GridSearchCV`  
- `RandomizedSearchCV`  
- `Bayesian Optimization` (Optuna, bayes_opt, etc.)

> Explique os parâmetros, mostre os melhores resultados e, se possível, visualize os scores.

---

### 4.5. Entrega

- **Prazo:** até **09/05/2025**
- **Envio via Canvas**
