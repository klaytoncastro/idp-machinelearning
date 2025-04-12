# Elaborando uma abordagem de Machine Learning para uma Tarefa de Classificação

## Descrição do Dataset

O dataset pode ser encontrado no seguinte link: [Dataset do Banco](https://github.com/bluenex/WekaLearningDataset/blob/master/bank/bank-data.csv). Ele contém informações sobre clientes de um banco e se eles adquiriram um produto de investimento pessoal (PEP). Os atributos incluem:

- **id**: Identificador único para cada cliente.
- **age**: Idade do cliente em anos.
- **sex**: Sexo do cliente ("FEMALE" ou "MALE").
- **region**: Região geográfica onde o cliente reside ("INNER_CITY", "TOWN", "RURAL", entre outras).
- **income**: Renda anual do cliente.
- **married**: Estado civil ("YES" para casado, "NO" para solteiro).
- **children**: Número de filhos.
- **car**: Posse de carro ("YES" ou "NO").
- **save_act**: Conta poupança ("YES" ou "NO").
- **current_act**: Conta corrente ("YES" ou "NO").
- **mortgage**: Hipoteca ("YES" ou "NO").
- **pep**: Produto de investimento pessoal adquirido ("YES" ou "NO").

## Análise Exploratória de Dados (EDA)

- Realize uma análise exploratória para entender melhor as características dos dados, distribuições e possíveis correlações entre as variáveis.

## Pré-processamento de Dados

1. **Tratamento de Valores Ausentes**: Verifique e trate valores ausentes.
2. **Codificação de Variáveis Categóricas**: Utilize técnicas como One-Hot Encoding ou Label Encoding.
3. **Normalização/Padronização**: Normalize ou padronize os dados numéricos para evitar viés devido à escala das variáveis.

## Seleção de Features e Modelagem

- Aplique diferentes algoritmos de machine learning como Regressão Logística, Árvores de Decisão, K-NN, SVM, e compare os resultados usando validação cruzada para robustez.

### Explorando padrões com abordagens não supervisionadas

> Será que clientes com perfis semelhantes se comportam de forma parecida na aquisição do PEP?  
> E se vocês tentassem **agrupar os clientes primeiro**, e usar esse grupo como **uma nova informação no modelo**?

- **Dica**: além dos métodos supervisionados tradicionais, você também pode utilizar **técnicas não supervisionadas**, como algoritmos de **clusterização (ex: K-Means)**, para identificar **grupos de clientes com perfis similares**.
- Esses agrupamentos podem revelar **padrões ocultos** nos dados e até serem utilizados como **novas variáveis ("labels artificiais")** para enriquecer os modelos de classificação.
- Essa estratégia é conhecida como **semi-supervised learning** ou **feature augmentation via clustering**, e pode melhorar o desempenho preditivo do modelo final.


<!--

## Otimização de Hiperparâmetros

- Utilize métodos como Grid Search, Random Search e/ou otimização Bayesiana para encontrar os melhores hiperparâmetros para os modelos escolhidos.

-->

## Interpretação do Modelo

- Interprete os resultados dos modelos analisando a importância das variáveis, matriz de confusão e métricas de avaliação como precisão, recall e F1-score.

Essa abordagem combinada de análise exploratória, pré-processamento, modelagem, otimização e interpretação fornece uma visão compreensiva sobre os fatores que influenciam a aquisição de um PEP pelos clientes do banco.