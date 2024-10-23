import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Definindo o dataset
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# Criando o codificador de transações e transformando o dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Executando o algoritmo Apriori para encontrar conjuntos de itens frequentes
frequent_itemsets = apriori(df, min_support=0.7, use_colnames=True)

# Gerando as regras de associação a partir dos conjuntos de itens frequentes
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Imprimindo as regras de associação encontradas
print(rules)

#!pip install pyfpgrowth
import pyfpgrowth
transactions = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
                ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
                ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
                ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
                ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
print(rules)