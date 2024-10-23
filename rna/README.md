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

<!--

## 3. Conceitos de Gradientes e Funções de Ativação

Antes da aplicação das funções de ativação, é importante entender o conceito de gradiente, uma medida de como uma função muda em relação às suas variáveis de entrada. Em linhas gerais, o gradiente indica a direção e a intensidade da mudança necessária nos parâmetros de um modelo (como os pesos de uma rede neural) para minimizar a função de custo (loss function), podendo ser usado no processo de otimização, onde ajustamos os pesos na direção que reduz o erro da rede, mecanismo conhecido como **Gradiente Descedente**. Aqui estão os tipos mais comuns de gradiente usados em machine learning e redes neurais:

### Gradiente Padrão (Batch Gradient)
   - O gradiente é calculado usando o conjunto completo de dados.
   - **Vantagem:** Produz uma atualização estável, já que considera todos os dados.
   - **Desvantagem:** Pode ser computacionalmente caro e lento para grandes volumes de dados.

### Gradiente Estocástico (Stochastic Gradient Descent - SGD)
   - O gradiente é calculado usando apenas uma amostra de dados por vez.
   - **Vantagem:** Rápido e eficiente para grandes conjuntos de dados, com atualizações frequentes.
   - **Desvantagem:** As atualizações podem ser ruidosas e menos estáveis, podendo pular o ótimo local ou global.

### Gradiente em Mini-Lote (Mini-Batch Gradient Descent)
   - O gradiente é calculado usando um pequeno grupo (ou mini-lote) de amostras de dados.
   - **Vantagem:** Compromisso entre estabilidade e velocidade, combinando a eficiência do SGD com a estabilidade do Batch Gradient.
   - **Desvantagem:** Ainda pode ter alguma variação nas atualizações, mas é menos ruidoso do que o SGD puro.

### Como o Gradiente é Calculado?

O cálculo do gradiente envolve derivadas, que medem a taxa de mudança da função de custo em relação aos parâmetros da rede. A **derivada parcial** de cada peso indica quanto o erro mudaria se fizéssemos uma pequena mudança nesse peso específico. O conjunto dessas derivadas parciais forma o vetor gradiente, que é usado para atualizar os pesos na direção que minimiza a função de custo.

## Vanishing Gradient e Exploding Gradient

Durante o treinamento de redes neurais, podem surgir dois problemas relacionados ao gradiente:

### 1. Vanishing Gradient (Gradiente que Desaparece)
   - Ocorre quando o gradiente fica muito pequeno durante o backpropagation, dificultando a atualização dos pesos. Isso é especialmente problemático em redes profundas, onde os gradientes diminuem exponencialmente ao retroceder pelas camadas.
   - **Impacto:** Treinamento lento ou estagnado, especialmente para as camadas iniciais da rede.

### 2. Exploding Gradient (Gradiente que Explode)
   - Ocorre quando o gradiente se torna muito grande, causando mudanças drásticas nos pesos da rede, o que pode levar a instabilidade no treinamento e à divergência do modelo.
   - **Impacto:** Pesos muito grandes, gerando resultados incoerentes e impedindo a convergência do modelo.

## Conclusão

Compreender o conceito de gradiente e seus tipos é essencial antes de explorar as funções de ativação. O gradiente é a ferramenta que a rede usa para ajustar seus pesos e melhorar a performance, e problemas como vanishing e exploding gradients influenciam diretamente a escolha das funções de ativação adequadas.


-->

### Principais Funções de Ativação

ntes da aplicação das funções de ativação, é importante entender o conceito de gradiente, uma medida de como uma função muda em relação às suas variáveis de entrada. Em linhas gerais, o gradiente indica a direção e a intensidade da mudança necessária nos parâmetros de um modelo (como os pesos de uma rede neural) para minimizar a função de custo (loss function), podendo ser usado no processo de otimização, onde ajustamos os pesos na direção que reduz o erro da rede, mecanismo conhecido como **Gradiente Descedente**. As funções de ativação são componentes essenciais das Redes Neurais Artificiais, pois introduzem não-linearidade nos modelos, permitindo que as redes aprendam padrões complexos nos dados. A seguir, estão os conceitos das principais funções de ativação: 

### Sigmoid
A função Sigmoid é uma função que comprime qualquer valor de entrada para um intervalo entre 0 e 1. Essa característica é ideal para problemas de classificação binária, onde você quer que a saída seja interpretada como uma probabilidade. A suavidade da função ajuda a modelar a transição entre diferentes classes de maneira gradual, tornando-a fácil de entender como uma probabilidade.

**Vantagem:** Ela torna as transições suaves e facilita a interpretação probabilística dos resultados.
**Desvantagem:** Um problema significativo é o "vanishing gradient" (gradiente que desaparece), onde os gradientes ficam muito pequenos durante o treinamento, o que atrapalha a atualização dos pesos da rede.

### Tanh (Tangente Hiperbólica)
A função Tanh é similar à Sigmoid, mas suas saídas variam de -1 a 1, em vez de 0 a 1. Isso significa que os valores estão centrados em zero, o que pode ajudar no treinamento da rede, permitindo que os pesos se ajustem mais eficientemente.

**Vantagem:** A saída centrada em zero pode levar a uma convergência mais rápida durante o treinamento.
**Desvantagem:** Assim como a Sigmoid, a Tanh também sofre com o problema do vanishing gradient, especialmente para valores de entrada extremos.

## ReLU (Rectified Linear Unit)

A ReLU é uma função simples e direta: qualquer valor negativo é convertido para zero, enquanto valores positivos permanecem os mesmos. Essa simplicidade é a razão pela qual ela é a função de ativação mais popular para redes profundas.

**Vantagem:** A ReLU resolve o problema do vanishing gradient, permitindo que gradientes maiores passem pela rede e acelerando o treinamento.
**Desvantagem:** Um problema que pode ocorrer é o "neurônio morto," onde certos neurônios podem parar de aprender (ficar sempre em zero) durante o treinamento, caso o gradiente seja zero para valores negativos.

### Leaky ReLU
A Leaky ReLU é uma variação da ReLU que permite que valores negativos passem, mas de forma reduzida (um pequeno valor negativo). Isso resolve parcialmente o problema de neurônios mortos, pois sempre há um pequeno gradiente mesmo para valores negativos.

**Vantagem:** Reduz o problema de neurônios mortos ao manter um pequeno gradiente para valores negativos, melhorando a aprendizagem.
**Desvantagem:** A escolha do valor do parâmetro para o gradiente negativo (geralmente um número pequeno, como 0.01) é arbitrária e pode não ser adequada para todos os problemas. Além disso, se esse valor for mal ajustado, a rede pode não aprender de forma eficaz, e valores negativos ainda podem impactar o aprendizado, mesmo que em menor grau.

### Softmax
A função Softmax é usada quando a saída do modelo precisa ser uma distribuição de probabilidades entre múltiplas classes, como em problemas de classificação multi-classe. Ela transforma as saídas de modo que todas as probabilidades somem exatamente 1.

**Vantagem:** A Softmax garante que a soma das probabilidades para todas as classes seja igual a 1, tornando mais fácil interpretar o resultado como uma probabilidade distribuída entre as classes possíveis.
**Desvantagem:** A Softmax pode ser sensível a valores extremos, fazendo com que pequenas mudanças nos dados de entrada causem grandes alterações nas probabilidades atribuídas às classes. Além disso, em redes neurais profundas, há um risco de sofrer com o problema do vanishing gradient, o que pode dificultar o treinamento. A Softmax também pressupõe que as classes são mutuamente exclusivas, o que nem sempre é o caso em problemas multi-rótulo.




## 4. Ferramentas: TensorFlow e Keras

Keras é uma biblioteca de código aberto de alto nível que fornece uma interface simplificada para a construção e treinamento de redes neurais. Ela foi projetada para facilitar o uso do TensorFlow, uma das plataformas mais poderosas e amplamente utilizadas para aprendizado de máquina e redes neurais. Keras é integrado ao TensorFlow como sua API principal de alto nível a partir da versão 2.0, com suporte robusto da Google.

Com o Keras, é possível construir protótipos rapidamente e testar diferentes arquiteturas de redes neurais sem a necessidade de escrever código extenso e detalhado. Isso acelera o processo de experimentação, essencial para pesquisadores e desenvolvedores, tornando-a uma escolha recomendada nesta disciplina. Embora o PyTorch também seja uma excelente biblioteca, amplamente utilizada em machine learning e deep learning, ele exige mais controle manual em alguns aspectos, como a definição explícita do ciclo de treinamento, o que pode complicar a implementação.

## 4. Principais Algoritmos e Comparação com Modelos Clássicos de ML

### Adaline 

O Adaline (ADAptive LINear Element) introduziu conceitos de erro contínuo e ajustamento de pesos baseado no erro. É um precursor dos modelos modernos e ajuda a entender a base do aprendizado supervisionado. 

### Perceptron

O perceptron fpo primeiro modelo que demonstrou ser capaz de aprender padrões lineares de forma supervisionada. Apesar de capaz de resolver problemas simples, suas limitações motivaram a evolução para um algoritmo que implementa redes Perceptron multicamadas (MLP).

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

**Nota**: A compressão em autoencoders pode ser comparada à técnica de PCA (Análise de Componentes Principais), vista em algoritmos de aprendizado não supervisionado. Contudo, o emprego de autoencoders é muito mais poderoso, sendo capaz de lidar com dados em espaços não lineares. Um Autoencoder é projetado para comprimir e reconstruir os dados, aprendendo uma representação latente dos dados sem supervisão (aprendizado não supervisionado)Segue um exemplo de implementação: 

```python
# Importando as bibliotecas necessárias
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

# Exemplo de dados de entrada (transações de compra)
# Neste caso, temos um conjunto de dados similar ao usado no MLP, mas o objetivo é comprimir e reconstruir os dados.
X = np.array([[1, 1, 0, 0],  # Exemplo de compras de um usuário
              [0, 1, 0, 1],  # Outro usuário
              [1, 1, 1, 0],
              [0, 0, 1, 1]])

# Definir a dimensão de entrada, que corresponde ao número de características dos dados
input_dim = X.shape[1]  # Neste caso, input_dim é 4 porque temos 4 características

# Construção do Autoencoder
# O autoencoder é composto por duas partes principais: Encoder (codificador) e Decoder (decodificador).

# Criando a camada de entrada com a dimensão dos dados
input_layer = Input(shape=(input_dim,))  # input_dim é o número de características de cada amostra

# Encoder: reduz a dimensionalidade dos dados para 2 neurônios
# A função de ativação 'relu' é usada para introduzir não linearidade
encoder = Dense(2, activation="relu")(input_layer)

# Decoder: reconstrói os dados para a dimensão original
# A função de ativação 'sigmoid' mapeia a saída para o intervalo [0, 1], útil se os dados forem normalizados
decoder = Dense(input_dim, activation="sigmoid")(encoder)

# Definindo o modelo Autoencoder completo
# Conectamos a camada de entrada ao decodificador para criar o modelo final
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compilando o modelo
# Otimizador 'adam' para ajustar os pesos e 'binary_crossentropy' para medir a perda entre a entrada e a saída reconstruída
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Treinando o modelo
# X é usado tanto como entrada quanto como saída, pois queremos que o autoencoder aprenda a reconstruir os dados
# Treinamos o modelo por 100 épocas; verbose=0 significa que não queremos ver os detalhes do progresso de treinamento
autoencoder.fit(X, X, epochs=100, verbose=0)

# Fazer previsões (reconstrução das entradas)
# Após o treinamento, o autoencoder tenta reconstruir as amostras de entrada
reconstructed = autoencoder.predict(X)
print("Dados originais:")
print(X)
print("\nDados reconstruídos:")
print(reconstructed)  # Exibe os dados reconstruídos, que devem ser próximos aos dados originais
```

### MLP (Multi Layer Perceptron)

O MLP é focado em prever resultados específicos com base nos dados de entrada (aprendizado supervisionado). É amplamente utilizado em diversas tarefas: 

- **Classificação**: Determinação de categorias para novos dados com base em características observadas (e.g., classificar um e-mail como spam ou não).
- **Regressão**: Predição de valores contínuos com base em dados de entrada (e.g., prever preços de imóveis).
- **Sistemas de Recomendação**: Previsão de itens que um usuário pode gostar com base em seu histórico e padrões de comportamento.
- **Reconhecimento de Padrões**: Identificação de padrões em dados complexos, como reconhecimento de voz e imagens.

O treinamento do MLP envolve ajustar os pesos das conexões entre os neurônios para minimizar a diferença entre a previsão da rede e o valor real. Isso é feito através de um processo iterativo de propagação direta (forward pass) e reversa (backward pass): 

- **Propagação Direta**: Os dados de entrada passam pela rede, camada por camada. Em cada neurônio, é realizada uma **soma ponderada** das entradas seguida pela aplicação de uma função de ativação. A saída final da rede é comparada com o valor real (label) para calcular o erro. 

- **Retro-propagação (Reversa)**: O erro calculado é propagado para trás, ajustando os pesos das conexões para reduzir esse erro. O algoritmo de **Gradiente Descendente** é usado para ajustar os pesos na direção que minimiza a função de custo. O processo é repetido várias vezes (épocas) até que o erro seja minimizado a um nível aceitável.

Segue um exemplo de implementação: 

```python
# Instalar Keras (TensorFlow) se ainda não estiver instalado
!pip install tensorflow

# Importando as bibliotecas necessárias
import numpy as np
from tensorflow.keras.models import Sequential   # Importa a classe Sequential para criar o modelo MLP
from tensorflow.keras.layers import Dense        # Importa a camada Dense para definir as camadas da rede

# Exemplo simples de dados de entrada (transações de compra)
# Cada linha do array X representa uma transação e cada coluna representa um item específico.
# Um valor de 1 indica que o item foi comprado, e 0 indica que não foi comprado.
X = np.array([[1, 1, 0, 0],  # Exemplo de compras de um usuário: comprou os dois primeiros itens
              [0, 1, 0, 1],  # Outro usuário: comprou o segundo e o quarto itens
              [1, 1, 1, 0],  # Comprou os três primeiros itens
              [0, 0, 1, 1]]) # Comprou os dois últimos itens

# A saída (y) é o que estamos tentando prever.
#
```

### Redes Neurais vs. Algortimos Clássicos de ML

As RNAs têm uma capacidade de modelar padrões complexos e não lineares que as diferenciam dos algoritmos clássicos de ML, como Árvores de Decisão, Random Forests e SVMs. As redes neurais profundas potencialmente capturam padrões mais complexos por meio de múltiplas camadas de processamento e funções de ativação não lineares, que seriam difíceis ou impossíveis de modelar com métodos tradicionais. 

As RNAs também oferecem a vantagem do aprendizado de características. São capazes de aprender automaticamente representações complexas dos dados, extraindo características relevantes de forma hierárquica. Em problemas de imagens, por exemplo, redes convolucionais (CNNs) identificam bordas, formas e padrões sem a necessidade de pré-processamento manual. Em contraste, algoritmos clássicos geralmente exigem uma engenharia de características mais elaborada para alcançar bons resultados.

Além disso, as redes neurais são altamente eficazes em termos de escalabilidade e dados massivos. Elas podem ser treinadas com grandes quantidades de dados usando GPUs, o que permite uma escalabilidade que algoritmos como SVM podem não alcançar de maneira eficiente. Técnicas como regularização e dropout ajudam as RNAs a lidar com grandes volumes de dados, evitando problemas de overfitting.

Outro ponto forte das RNAs é sua capacidade adaptativa. Redes neurais podem ser ajustadas para diferentes tipos de dados e problemas, desde séries temporais, texto e imagens até jogos e simulações complexas. Algoritmos clássicos, por outro lado, tendem a ser mais específicos e limitados a certos tipos de problemas, o que torna as RNAs mais versáteis e aplicáveis a uma ampla gama de cenários.

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


- Use o código abaixo para importar o dataset: 

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