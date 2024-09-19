# Otimização de Hiperparâmetros em Machine Learning

## Objetivo

Nesta tarefa, cada estudante ou dupla deverá implementar os três métodos de otimização a seguir, buscando o melhor desempenho possível do algoritmo atribuído:

1. **Grid Search**: Realiza uma busca exaustiva testando todas as combinações possíveis de hiperparâmetros dentro de uma grade predefinida.
2. **Randomized Search**: Realiza uma busca aleatória, explorando combinações de hiperparâmetros de forma mais eficiente que o Grid Search para espaços de busca maiores.
3. **Otimização Bayesiana**: Utiliza modelos probabilísticos para guiar a busca pelos melhores hiperparâmetros, aproveitando informações de iterações anteriores para realizar uma busca mais inteligente.

Cada estudante ou dupla deverá otimizar um modelo de machine learning utilizando os três métodos clássicos de otimização de hiperparâmetros apresentados em sala de aula: **Grid Search**, **Randomized Search** e **Otimização Bayesiana**. O objetivo é comparar o desempenho dos modelos com diferentes configurações de hiperparâmetros e avaliar o impacto desta parametrização nas métricas de performance.  

## Algoritmos

Cada estudante ou dupla será responsável por um dos seguintes algoritmos, que foram previamente sorteados em sala de aula, e deverá(ão) realizar a otimização de seus respectivos hiperparâmetros na tarefa de classificação de vinhos do dataset [Wine Quality](https://github.com/klaytoncastro/idp-machinelearning/blob/main/optimization/winequality_ml_classifier_optimized.ipynb):

# Alocações dos Trabalhos em Dupla

# Alocações dos Trabalhos em Dupla

### 1. Decision Tree
Árvore de Decisão é um algoritmo de aprendizado supervisionado usado tanto para classificação quanto para regressão. Ele constrói um modelo de árvore a partir de dados de entrada, realizando decisões baseadas em características e seus valores.

- Felipe Pereira Dutra
- Kelwin dos Santos Menezes

---

### 2. Random Forest
Random Forest (Floresta Aleatória) é um conjunto de árvores de decisão construídas com diferentes subconjuntos de dados e características, melhorando a precisão e reduzindo o risco de overfitting.

- Sara Pacheco de Azevedo
- Fábio Luís de Carvalho Terra

---

### 3. Extra Tree
Extra Tree (Extremely Randomized Tree) é uma variante do algoritmo de Árvore de Decisão, onde as divisões em cada nó da árvore são escolhidas de forma completamente aleatória, resultando em maior variabilidade entre as árvores.

- Pedro Calil Raposo Mingossi Cordeiro
- João Gabriel Gonçalves Oliveira

---

### 4. Extra Trees
Extra Trees (Extremely Randomized Trees) é uma variante do Random Forest, que cria várias árvores de decisão altamente randomizadas, aumentando a robustez e precisão da floresta.

- Arthur Torquato Novaes
- Felipe Barroso de Castro

---

### 5. XGBoost
XGBoost (Extreme Gradient Boosting) é um algoritmo de boost de gradiente eficiente e otimizado que usa uma sequência de árvores de decisão para melhorar o desempenho preditivo.

- Eduardo Milagres Lima
- Igor Caldeira Andrade

---

### 6. LightGBM
LightGBM (Light Gradient Boosting Machine) é um algoritmo de boost de gradiente focado em eficiência e velocidade, particularmente eficaz para grandes volumes de dados.

- Luca Verdade Lenzoni
- Lucas Fiche Ungarelli Borges

---

### 7. Naive Bayes
Naive Bayes é um algoritmo baseado na Teoria de Probabilidade de Bayes, assumindo que todas as características são independentes entre si, usado principalmente para problemas de classificação.

- Claudio da Aparecida Meireles Filho
- Pedro Henrique Pontes Fontana

---

### 8. Logistic Regression
Logistic Regression (Regressão Logística) é um algoritmo de classificação que modela a probabilidade de uma variável binária, muito usado para classificação binária.

- Mariana Magalhaes Silva
- Leonardo Freitas Barboza

---

### 9. SVM (Support Vector Classifier - SVC)
SVM (Support Vector Machine) é um algoritmo de aprendizado supervisionado que tenta encontrar um hiperplano que separa os dados em diferentes classes com a máxima margem possível.

- Lucas Fidalgo Bitencourt
- Mateus Batista Peixoto da Silva

---

### 10. k-NN (k-Nearest Neighbors)
k-NN (k-Nearest Neighbors - k-Vizinhos Mais Próximos) é um algoritmo simples de classificação que classifica os dados com base nas classes dos vizinhos mais próximos no espaço de características.

- João Henrique de Oliveira Salles
- Lucas Narita Nunes de Melo Freita

---


## Técnicas de Otimização



## Recursos Necessários

- **Slides e Notebooks de Exemplo**: Disponíveis no repositório [Notas de Aula](https://github.com/klaytoncastro/idp-machinelearning/blob/main/optimization/ML_Optimization.pdf) e nos arquivos `.ipynb` de otimização.
- **Documentação do scikit-learn**: [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- **Documentação do XGBoost**: [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- **Documentação do LightGBM**: [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)

## Métricas de Avaliação

Os modelos devem ser avaliados utilizando as seguintes métricas:

- **Acurácia**: Percentual de previsões corretas sobre o total de previsões.
- **Precisão**: Proporção de verdadeiros positivos sobre o total de positivos previstos.
- **Recall**: Proporção de verdadeiros positivos sobre o total de positivos reais.
- **F1-Score**: Média harmônica entre precisão e recall, equilibrando ambos.

## Formato do Relatório
Cada estudante ou dupla deverá documentar o processo de otimização, com as seguintes seções:

1. **Introdução ao Algoritmo**: Descrição do algoritmo utilizado e de seus principais hiperparâmetros.
2. **Metodologia**: Explicação das técnicas de otimização e hiperparâmetros testados. Justifique a escolha das métricas usadas para otimização.
3. **Resultados**: Apresente uma comparação dos resultados das três técnicas de otimização e seus respectivos desempenhos. Inclua gráficos, tabelas e discussões sobre os tempos de execução e a efetividade de cada método.
4. **Discussão**: Reflexões sobre os resultados, com possíveis melhorias para otimização e sobre o impacto da escolha dos hiperparâmetros.

## Apresentação

Os resultados deverão ser apresentados em sala de aula, com o uso de gráficos e tabelas que ilustrem o impacto das diferentes técnicas de otimização sobre o desempenho do modelo. 
