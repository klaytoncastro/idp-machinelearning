## Apriori 

O algoritmo Apriori (via biblioteca mlxtend) é um clássico para mineração de regras de associação. Ele calcula padrões frequentes e regras como confiança e suporte.

**Grupo 3:** Lucas Fiche, Cláudio, Leonardo, Igor

```python
# Instalar mlxtend
!pip install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Exemplo simples de dataset de transações
dataset = pd.DataFrame({
    'Leite': [1, 1, 0, 1, 1],
    'Pão': [1, 1, 1, 0, 1],
    'Manteiga': [0, 1, 0, 1, 0],
    'Cerveja': [1, 0, 0, 1, 0],
    'Refrigerante': [0, 0, 1, 1, 1]
})

# Gerar itemsets frequentes com um suporte mínimo de 50%
frequent_itemsets = apriori(dataset, min_support=0.5, use_colnames=True)

# Gerar regras de associação a partir dos itemsets frequentes
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Exibir itemsets e regras
print(frequent_itemsets)
print(rules)
```

## FP-Growth 

**Grupo 4**: Mariana, Eduardo, João Gabriel, Pedro Calil

FP-Growth (via biblioteca mlxtend) é uma alternativa potencialmente mais eficiente ao Apriori, pois evitando a geração explícita de candidatos ele se torna eficiente mesmo em conjuntos de dados maiores.

```python
# Instalar mlxtend (se ainda não instalou)
!pip install mlxtend

from mlxtend.frequent_patterns import fpgrowth

# Usando o mesmo dataset
frequent_itemsets_fp = fpgrowth(dataset, min_support=0.5, use_colnames=True)

# Gerar regras de associação a partir dos itemsets frequentes
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.7)

# Exibir itemsets e regras
print(frequent_itemsets_fp)
print(rules_fp)
```

<!--
https://archive.ics.uci.edu/dataset/352/online+retail
-->

## Importação do Dataset

| Variável       | Descrição                                                                 |
|----------------|---------------------------------------------------------------------------|
| **InvoiceNo**  | Número da fatura. Se este código começar com a letra 'c', indica um cancelamento. |
| **StockCode**  | Código do produto (único por produto).                                    |
| **Description**| Nome do produto.                                                         |
| **Quantity**   | Quantidade de cada produto por transação.                                 |
| **InvoiceDate**| Data e hora da fatura.                                                    |
| **UnitPrice**  | Preço unitário do produto.                                                |
| **CustomerID** | Código do cliente (único por cliente).                                    |
| **Country**    | País de onde o cliente fez a compra.                                      |


```python
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
online_retail = fetch_ucirepo(id=352) 
  
# data (as pandas dataframes) 
X = online_retail.data.features 
y = online_retail.data.targets 
  
# metadata 
print(online_retail.metadata) 
  
# variable information 
print(online_retail.variables) 
```

<!--

-->