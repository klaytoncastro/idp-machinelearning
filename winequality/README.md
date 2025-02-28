# Desafio 01 - Wine Quality
<!-- e [Regressão](./winequality_ml_regressor.ipynb) -->

## Tarefa de Classificação

Para começar nossa jornada, vamos explorar uma das tarefas fundamentais da área: a **classificação**. Essa tarefa consiste em ensinar um modelo a identificar e categorizar dados com base em padrões aprendidos a partir de exemplos (Aprendizado Supervisionado).

Para isso, utilizaremos o Wine Quality Dataset, que contém informações físico-químicas de amostras de vinho, além de uma avaliação de qualidade por especialistas. Nosso objetivo é construir um modelo de classificação supervisionada capaz de prever a qualidade do vinho a partir de suas propriedades físico-químicas.

## Como executar? 

Baixe o notebook com o exemplo de [Classificação](./winequality_ml_classifier.ipynb) e execute-o passo a passo em seu ambiente Jupyter ou Google Colab para compreender como abordamos as tarefas de classificação de vinhos em brancos ou tintos e a previsão de sua qualidade (nota). 

## Descrição das Variáveis

O conjunto de dados **Wine Quality** contém características físico-químicas de diferentes amostras de vinho e sua respectiva avaliação de qualidade. Abaixo está a descrição de cada variável presente no dataset:

### Variáveis Independentes (Características Físico-Químicas)

1. **fixed acidity** (*acidez fixa*) - [float]  
   - Representa os ácidos não voláteis, como ácido tartárico. Está relacionada ao frescor e sabor do vinho.

2. **volatile acidity** (*acidez volátil*) - [float]  
   - Medida da quantidade de ácido acético no vinho, que pode impactar negativamente o sabor (excesso leva a gosto avinagrado).

3. **citric acid** (*ácido cítrico*) - [float]  
   - Presente em pequenas quantidades, adiciona frescor e melhora a preservação do vinho.

4. **residual sugar** (*açúcar residual*) - [float]  
   - Quantidade de açúcar remanescente após a fermentação. Pode afetar o dulçor e o corpo do vinho.

5. **chlorides** (*cloretos*) - [float]  
   - Representa o teor de sal no vinho, que pode influenciar no sabor.

6. **free sulfur dioxide** (*dióxido de enxofre livre*) - [float]  
   - Atua como conservante, protegendo contra oxidação e bactérias.

7. **total sulfur dioxide** (*dióxido de enxofre total*) - [float]  
   - Soma do SO₂ livre e ligado. Em excesso, pode causar aromas desagradáveis.

8. **density** (*densidade*) - [float]  
   - Relacionada à concentração de açúcar, álcool e outros compostos. Impacta na percepção do corpo do vinho.

9. **pH** (*pH*) - [float]  
   - Mede a acidez geral do vinho. Valores comuns variam entre **2,7 e 4,0**.

10. **sulphates** (*sulfatos*) - [float]  
    - Contribui para a estabilidade microbiológica e realce do sabor.

11. **alcohol** (*teor alcoólico*) - [float]  
    - Influencia diretamente o corpo e a sensação do vinho.

### Variável Alvo 1 (Cor do Vinho)

12. **color** (*cor do vinho*) - [string]  
    - Identifica se o vinho é **tinto** ou **branco** (em algumas versões do dataset, essa informação está separada).

### Variável Alvo 2 (Qualidade do Vinho)

13. **quality** (*qualidade do vinho*) - [int]  
    - Avaliação sensorial dada por especialistas, variando de **0 a 10** (geralmente entre **3 e 9** no dataset).

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

Com essa abordagem, conseguimos explorar tanto os conceitos fundamentais de estatística descritiva quanto os princípios básicos da classificação em Machine Learning. À medida que avançarmos no curso, aprofundaremos nossas análises, explorando técnicas mais sofisticadas para otimização dos modelos.