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

## Tarefa

Este é um dataset que contém todas as transações ocorridas entre 01/12/2010 e 09/12/2011 para um varejista online britânico. A empresa vende presentes para várias ocasiões e muitos de seus clientes são atacadistas.

<!--
https://archive.ics.uci.edu/dataset/352/online+retail
-->

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

Cada um dos grupos deverá usar o `Apriori` ou  `FP-Growth` para identificar padrões de compra frequentes entre os produtos e apresentar em sala sua implementação. 

### Importação do Dataset

```python
import pandas as pd
import requests
import zipfile
import io

# URL do arquivo
url = "https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/resources/online_retail.zip"

# Fazendo o download e descompactando o arquivo zip em memória
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open('online_retail_dataset.csv') as f:
        df = pd.read_csv(f)

# Exibindo as primeiras linhas do DataFrame
print(df.head())
```
