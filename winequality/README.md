# Desafio 01 - Wine Quality

## Origem do Dataset e Contexto da Pesquisa

O conjunto de dados **Wine Quality** foi introduzido no artigo científico intitulado  
**"Modeling wine preferences by data mining from physicochemical properties"**, publicado em **2009** na revista *Decision Support Systems*. O estudo de Cortez et al. (2009) utilizou mineração de dados para prever a qualidade do vinho "Vinho Verde" branco e tinto de Portugal, com base em testes físico-químicos disponíveis na etapa de certificação. Diferentes técnicas de Machine Learning foram aplicadas para modelar essa relação, permitindo apoiar avaliações sensoriais de enólogos e otimizar a produção de vinho. O modelo desenvolvido demonstrou potencial para complementar as avaliações sensoriais dos enólogos, fornecendo suporte na tomada de decisões e otimizando a produção de vinho.

## Tarefa de Classificação

Para começar nossa jornada, vamos explorar uma das tarefas fundamentais da área: a **classificação**. Essa tarefa consiste em ensinar um modelo a identificar e categorizar dados com base em padrões aprendidos a partir de exemplos (Aprendizado Supervisionado).

Utilizaremos o Wine Quality Dataset, que contém informações físico-químicas de amostras de vinho e suas respectivas avaliações de qualidade por especialistas. O desafio consiste em desenvolver um modelo de classificação supervisionada para prever a qualidade do vinho com base nessas características.

## Como executar? 

Baixe o notebook com o exemplo de [Classificação](./winequality_ml_classifier.ipynb) e execute-o passo a passo em seu ambiente Jupyter ou Google Colab. Nele, você aprenderá como abordar a classificação dos vinhos (brancos ou tintos) e a previsão de sua qualidade (nota).

## Descrição das Variáveis

O conjunto de dados **Wine Quality** contém características físico-químicas de diferentes amostras de vinho e sua respectiva avaliação de qualidade. Abaixo está a descrição de cada variável presente no dataset:

### Variáveis Independentes (Características Físico-Químicas)

| **Variável**            | **Tipo**  | **Descrição** |
|-------------------------|----------|--------------|
| `fixed acidity`        | *float*  | Representa os ácidos não voláteis, como ácido tartárico. Relaciona-se ao frescor e sabor do vinho. |
| `volatile acidity`     | *float*  | Quantidade de ácido acético no vinho. Excesso pode causar sabor avinagrado. |
| `citric acid`         | *float*  | Adiciona frescor e melhora a preservação do vinho. |
| `residual sugar`      | *float*  | Açúcar remanescente após a fermentação, afeta dulçor e corpo do vinho. |
| `chlorides`           | *float*  | Representa o teor de sal, podendo influenciar no sabor. |
| `free sulfur dioxide` | *float*  | Atua como conservante, protegendo contra oxidação e bactérias. |
| `total sulfur dioxide`| *float*  | Soma do SO₂ livre e ligado. Excesso pode causar aromas desagradáveis. |
| `density`            | *float*  | Relacionada à concentração de açúcar, álcool e outros compostos. Impacta na percepção do corpo do vinho. |
| `pH`                 | *float*  | Mede a acidez geral do vinho. Valores comuns variam entre **2,7 e 4,0**. |
| `sulphates`          | *float*  | Contribui para a estabilidade microbiológica e realce do sabor. |
| `alcohol`            | *float*  | Influencia diretamente o corpo e a sensação do vinho. |

### Variáveis Alvo (Previsão na Modelagem)

| **Variável** | **Tipo**  | **Descrição** |
|-------------|----------|--------------|
| `color`     | *string* | Classificação do vinho em tinto ou branco. |
| `quality`   | *int*    | Avaliação sensorial por especialistas, variando de **0 a 10** (geralmente entre **3 e 9** no dataset). |

---

## Etapas da Análise de Dados e Modelagem Preditiva

- 1. Exploração do dataset – Compreensão dos dados, estatísticas descritivas e visualização.
- 2. Pré-processamento – Limpeza, normalização e engenharia de features.
- 3. Treinamento de um modelo de classificação – Utilizando algoritmos como árvores de decisão, random forest e outros.
- 4. Avaliação do modelo – Introdução a métricas como acurácia, precisão, recall e matriz de confusão.
- 5. Interpretação dos resultados – Análise do impacto das variáveis na classificação.

## Conclusão

Ao longo desta primeira tarefa, estabelecemos uma base sólida para compreendermos os desafios e aplicações da classificação em Machine Learning, preparando o terreno para problemas mais complexos ao longo do curso.

Antes de abordarmos a modelagem preditiva, enfatizamos a importância da estatística descritiva, analisando as características de cada variável e seu impacto no Wine Quality Dataset. Exploramos:

- Medidas de tendência central (média, mediana) e dispersão (desvio padrão, intervalo interquartil).
- Distribuição das variáveis (contagem, valores únicos, detecção de outliers via histogramas e boxplots).
- Relação entre as variáveis preditoras e a variável alvo (quality).
- Utilizamos as bibliotecas Pandas e Numpy para manipulação de dados, Seaborn/Matplotlib para visualização e Scikit-Learn para construção dos modelos.

Após carregar e explorar o dataset, realizamos as seguintes etapas:

### Análise Exploratória:

- Obtivemos uma visão geral do dataset (.describe() e .info()).
- Identificamos padrões e valores ausentes.


### Análise Estatística e Visualização:

- Exploramos a distribuição das variáveis numéricas via histogramas.
- Utilizamos boxplots para detectar outliers e avaliar seu impacto no modelo.
- Contamos valores das variáveis categóricas (color) e discretas (quality).


### Modelagem Preditiva:

- Primeiramente, previmos a cor do vinho (tinto ou branco), um problema de classificação.
- Em seguida, previmos sua qualidade (nota) com base em suas características químicas.
- Utilizamos o algoritmo `ExtraTrees`, um modelo baseado em árvores de decisão, devido à sua capacidade de lidar com conjuntos de dados complexos e identificar automaticamente a importância das variáveis na predição.

Concluímos essa primeira tarefa explorando os fundamentos de estatística descritiva e classificação em Machine Learning.
Nos próximos passos, vamos aprofundar a análise e otimizar os modelos, utilizando técnicas mais avançadas para aumentar a precisão das previsões!

<!-- e [Regressão](./winequality_ml_regressor.ipynb)  

### Referências adicionais

- [Acesso ao Artigo no CMU StatLib](http://lib.stat.cmu.edu/datasets/)
- [Dataset Original no Repositório do UCI Machine Learning](https://archive.ics.uci.edu/dataset/186/wine%2Bquality)

Os resultados indicaram que a **máquina de vetor de suporte (SVM)** obteve desempenho superior em comparação com os métodos de **regressão múltipla e redes neurais**. 

### 📌 **Referência completa do artigo**
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).  
**Modeling wine preferences by data mining from physicochemical properties**.  
*Decision Support Systems*, **47(4), 547-553**.  
🔗 
- [Artigo no CMU StatLib](http://lib.stat.cmu.edu/datasets/)

- [Acesso ao Artigo no ResearchGate](https://www.researchgate.net/publication/228342091_Modeling_wine_preferences_by_data_mining_from_physicochemical_properties) 
- [Página do Autor Paulo Cortez](https://www3.dsi.uminho.pt/pcortez/wine/)

-->
