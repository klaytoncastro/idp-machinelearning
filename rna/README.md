# RNA - Redes Neurais Artificiais

# 1. Visão Geral

Deep Learning é uma subárea do Machine Learning que se concentra em algoritmos inspirados na estrutura e funcionamento do cérebro humano, conhecidos como Redes Neurais Artificiais (RNAs). Elas são compostas por uma ou várias camadas de neurônios artificiais. De forma simplificada, cada neurônio recebe informações, realiza cálculos e o resultado pode ser utilizado pelos próximos neurônios, formando uma rede de conexões.

O termo "Deep" (profundo) refere-se ao uso de várias camadas escondidas entre a camada de entrada (dados iniciais) e a camada de saída (resultado final), formando o que é conhecido como redes neurais profundas. Quanto mais camadas, mais complexa e sofisticada é a capacidade da rede de aprender padrões não lineares complexos nos dados. Essas redes são amplamente usadas em tarefas que envolvem grandes volumes de dados, como reconhecimento de voz, visão computacional, processamento de linguagem natural e jogos. Essas aplicações geralmente exigem a identificação de padrões complexos que podem ser difíceis de capturar com abordagens tradicionais de estatística multivariada, mineração de dados e aprendizado de máquina.

As RNAs são modeladas de maneira semelhante ao funcionamento dos neurônios biológicos, mas com cálculos matemáticos que simulam a transmissão de informações. Elas são treinadas ajustando pesos nas conexões entre neurônios para minimizar o erro na previsão ou classificação de dados. Esse ajuste é feito iterativamente usando métodos como a retropropagação (backpropagation), que calcula os erros e ajusta os pesos para melhorar o desempenho do modelo, por meio da minimização da função de custo (loss function), que mede a discrepância entre a saída da rede e os valores reais esperados. Assim, os pesos das conexões entre neurônios podem ser ajustados iterativamente. 

## 2. Modelagem de Neurônios: Fundamentos e Técnicas de Aprendizado

- Neurônio MCP (McCulloch-Pitts): Um dos primeiros modelos de um neurônio artificial, que introduz o conceito de soma ponderada de entradas seguido por uma função matemática para ativação.

- Regra de Hebb: Baseado no princípio de que "neurônios que disparam juntos, se conectam juntos", apresenta o conceito de ajuste de força das conexões entre neurônios, essencial para compreender a formação de redes e o aprendizado de padrões.

- Neurônio Perceptron: O primeiro modelo de neurônio artificial capaz de aprender padrões linearmente separáveis como, por exemplo, a modelagem das portas lógicas `AND`e `OR`. Utiliza uma função de ativação simples para decidir a saída com base na soma ponderada das entradas e, em um espaço bidimensional, é possível traçar uma linha reta (hiperplano) que separa os exemplos positivos dos negativos. 

- Redes MLP: Perceptron Multicamadas resolve a limitação do perceptron simples, permitindo aprender padrões não lineares, com uso de funções de ativação como `sigmoid`, `tanh`, `softmax` e `relu`, tornando-o bastante relevante para tarefas de classificação e regressão. O MLP resolve a modelagem da porta lógica `XOR`, ao introduzir camadas ocultas com funções de ativação não lineares. A não linearidade das funções de ativação, como `relu` ou `sigmoid`, permite que o modelo aprenda padrões complexos, superando a limitação do perceptron simples. 

## 3. Ferramentas: TensorFlow e Keras

Keras é uma biblioteca de código aberto de alto nível que fornece uma interface simplificada para a construção e treinamento de redes neurais. Ela foi projetada para facilitar o uso do TensorFlow, uma das plataformas mais poderosas e amplamente utilizadas para aprendizado de máquina e redes neurais. Keras é integrado ao TensorFlow como sua API principal de alto nível a partir da versão 2.0, com suporte robusto da Google.

Com o Keras, é possível construir protótipos rapidamente e testar diferentes arquiteturas de redes neurais sem a necessidade de escrever código extenso e detalhado. Isso acelera o processo de experimentação, essencial para pesquisadores e desenvolvedores, tornando-a uma escolha recomendada nesta disciplina. Embora o PyTorch também seja uma excelente biblioteca, amplamente utilizada em machine learning e deep learning, ele exige mais controle manual em alguns aspectos, como a definição explícita do ciclo de treinamento, o que pode complicar a implementação.

## 4. Principais Algoritmos

### Adaline (ADAptive LINear Element) 

Introduz conceitos de erro contínuo e ajustamento de pesos baseado no erro. É um precursor dos modelos modernos e ajuda a entender a base do aprendizado supervisionado.

### Perceptron

O primeiro modelo que demonstrou ser capaz de aprender padrões lineares de forma supervisionada. Ele é importante para entender as limitações de modelos simples e motivou a evolução para redes multicamadas.

### Autoencoder 

Um autoencoder é uma rede neural mais robusta, treinada para copiar a entrada para a saída, composta por duas partes principais:

- Codificador (Encoder): Reduz a dimensionalidade dos dados, mapeando-os para um espaço de menor dimensão (representação comprimida).
- Decodificador (Decoder): Reconstrói os dados originais a partir dessa representação comprimida.
A compressão ocorre na camada intermediária, onde a rede aprende uma representação simplificada dos dados. 

Autoencoders são comumente aplicados para:

- Redução de dimensionalidade: Diminui o número de variáveis, preservando informações relevantes.
- Detecção de anomalias: Reconstruindo dados "normais", eles identificam padrões incomuns em dados anômalos.
- Compressão de imagens: Reduz o tamanho de arquivos sem perda significativa de qualidade.
- Extração de features: Cria uma representação compacta dos dados (atributos, variáveis ou características) para uso em outros modelos de machine learning.
- Sistemas de recomendação: Reconstrói entradas e prevê o próximo item com base em padrões ocultos.

**Nota**: A compressão em autoencoders pode ser comparada à técnica de PCA (Análise de Componentes Principais), vista em algoritmos de aprendizado não supervisionado. Contudo, o emprego de autoencoders é muito mais poderoso, sendo capaz de lidar com dados em espaços não lineares.

- Exemplo de implementação: 

```python
# Instalar Keras (TensorFlow) se ainda não estiver instalado
!pip install tensorflow

# Importando as bibliotecas necessárias
import numpy as np
from tensorflow.keras.models import Sequential   # Importa a classe Sequential para criar o modelo
from tensorflow.keras.layers import Dense        # Importa a camada Dense para criar as camadas do MLP

# Exemplo simples de dados de entrada (transações de compra)
# Cada linha representa uma transação e cada coluna um item (1 = comprado, 0 = não comprado)
X = np.array([[1, 1, 0, 0],  # Exemplo de compras de um usuário
              [0, 1, 0, 1],  # Outro usuário
              [1, 1, 1, 0],
              [0, 0, 1, 1]])

# A saída (y) poderia ser um item recomendado com base nas compras anteriores.
# Aqui estamos tentando prever se o usuário comprará um determinado item (1 = sim, 0 = não)
y = np.array([[1], [0], [1], [1]])  # Indicando se, por exemplo, "Leite" será comprado

# Criando a RNA (MLP) com a classe Sequential
# A rede é composta por uma camada de entrada, uma ou mais camadas ocultas, e uma camada de saída

# Camada de entrada com 4 neurônios, correspondente a 4 características de entrada
# A função de ativação 'relu' é usada para introduzir não linearidade
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))  # Camada de entrada com 8 neurônios

# Camada intermediária com 4 neurônios e função de ativação 'relu'
# Ela ajuda a capturar padrões complexos nos dados
model.add(Dense(4, activation='relu'))

# Camada de saída com 1 neurônio, usada para prever se o item será comprado (0 ou 1)
# A função de ativação 'sigmoid' limita a saída ao intervalo [0, 1], adequada para classificação binária
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
# 'binary_crossentropy' é a função de perda usada para problemas de classificação binária
# Otimizador 'adam' ajusta os pesos para minimizar a função de perda
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
# Treinamos o modelo com os dados de entrada (X) e as saídas correspondentes (y)
# Número de épocas = 100, verbose=0 para não exibir o progresso durante o treinamento
model.fit(X, y, epochs=100, verbose=0)

# Fazer previsões com o modelo treinado
# A previsão indicará a probabilidade de o item ser comprado com base nos dados de entrada
predictions = model.predict(X)
print(predictions)  # Exibe a probabilidade prevista para cada amostra de entrada
```

## 5. Atividade Prática

Cada grupo deverá implementar Redes Neurais Artificiais (RNA) com Autoencoders e MLP (Perceptron Multicamadas) para explorar padrões de compra e realizar previsões com base nas transações registradas. Utilizar o Google Colab é uma boa alternativa aqui. Ele disponibiliza GPUs e já tem o Keras instalado. 

### Autoencoder:

**Grupo 1**: Fábio, Sara, Arthur, Felipe Barroso, Lucas Narita
**Objetivo**: Utilizar um Autoencoder para compressão dos dados e extração de características.
**Tarefa**: Comprimir as características transacionais e usar essas representações para prever o próximo item com base em padrões ocultos nos dados.

### MLP (Perceptron Multicamadas):

**Grupo 2:** Felipe Dutra, Luca, Kelwin, Pedro 
**Objetivo**: Treinar uma rede neural MLP para prever ou classificar com base nas características dos clientes ou transações.
**Tarefa**: Implementar uma MLP para prever compras futuras ou categorizar transações com base em variáveis como preço, quantidade e país.

### Importação do Dataset

<!--
https://archive.ics.uci.edu/dataset/352/online+retail
-->

Este é um dataset que contém todas as transações ocorridas entre 01/12/2010 e 09/12/2011 para um varejista online britânico. A empresa vende presentes para várias ocasiões e muitos de seus clientes são atacadistas.


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


