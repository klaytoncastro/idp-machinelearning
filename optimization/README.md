# Otimização de Hiperparâmetros em Machine Learning

## Objetivo

Nesta tarefa, cada estudante ou dupla deverá implementar os três métodos de otimização a seguir, buscando o melhor desempenho possível do algoritmo atribuído:

1. **Grid Search**: Realiza uma busca exaustiva testando todas as combinações possíveis de hiperparâmetros dentro de uma grade predefinida.
2. **Randomized Search**: Realiza uma busca aleatória, explorando combinações de hiperparâmetros de forma mais eficiente que o Grid Search para espaços de busca maiores.
3. **Otimização Bayesiana**: Utiliza modelos probabilísticos para guiar a busca pelos melhores hiperparâmetros, aproveitando informações de iterações anteriores para realizar uma busca mais inteligente.

Cada estudante ou dupla deverá otimizar um modelo de machine learning utilizando os três métodos clássicos de otimização de hiperparâmetros apresentados em sala de aula: **Grid Search**, **Randomized Search** e **Otimização Bayesiana**. O objetivo é comparar o desempenho dos modelos com diferentes configurações de hiperparâmetros e avaliar o impacto desta parametrização nas métricas de performance.  

## Algoritmos

Cada estudante ou dupla será responsável por um dos seguintes algoritmos, que foram previamente sorteados em sala de aula, e deverá(ão) realizar a otimização de seus respectivos hiperparâmetros na tarefa de classificação de vinhos do dataset [Wine Quality](https://github.com/klaytoncastro/idp-machinelearning/blob/main/optimization/winequality_ml_classifier_optimized.ipynb):

- **Decision Tree**: Felipe e Kelvin
- **Random Forest**: Sara e Fábio
- **Extra Tree**: Pedro Calil e João Gabriel
- **Extra Trees**: Arthur e Felipe Castro
- **XGBoost**: Eduardo e Igor
- **LightGBM**: Luca Verdade e Lucas Fiche
- **Naive Bayes**: Cláudio e Pedro
- **Logistic Regression**: Mariana
- **SVM (Support Vector Classifier - SVC)**: Lucas Fidalgo e Mateus Batista
- **k-NN (k-Nearest Neighbors)**: ???

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
