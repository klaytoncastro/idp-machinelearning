# RNA - Redes Neurais Artificiais

Utilizar o Google Colab é uma boa alternativa aqui. Ele disponibiliza GPUs e já tem o Keras instalado. 

## Autoencoder 
Um Autoencoder pode ser usado para recomendação,  reconstruir entradas e prever o próximo item a ser comprado com base em padrões ocultos.

```python
# Importando as bibliotecas necessárias
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Definir a dimensão de entrada (quantidade de itens)
input_dim = X.shape[1]  # 4 itens de entrada no dataset

# Construir o Autoencoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(3, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

# Definir o modelo
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compilar o modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Treinar o modelo
autoencoder.fit(X, X, epochs=100, verbose=0)

# Fazer previsões (reconstrução das entradas)
reconstructed = autoencoder.predict(X)
print(reconstructed)
```


## MLP (Multilayer Perceptron) 

Usando o algoritmo Multilayer Perceptron (MLP) para Recomendação Simples, podemos  prever itens que um usuário poderia comprar, com base em uma simples rede neural. 


```python
# Instalar Keras (TensorFlow)
!pip install tensorflow

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Exemplo simples de dados (entrada: transações de compra, saída: item a ser recomendado)
# 1 = comprado, 0 = não comprado
X = np.array([[1, 1, 0, 0],  # Exemplo de compras de um usuário
              [0, 1, 0, 1],  # Outro usuário
              [1, 1, 1, 0],
              [0, 0, 1, 1]])

# A saída poderia ser o próximo item recomendado, por exemplo:
y = np.array([[1], [0], [1], [1]])  # Indicando se "Leite" (por exemplo) será comprado

# Criando a RNA (MLP)
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))  # Camada de entrada com 4 neurônios
model.add(Dense(4, activation='relu'))  # Camada intermediária
model.add(Dense(1, activation='sigmoid'))  # Saída binária (vai comprar ou não)

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X, y, epochs=100, verbose=0)

# Fazer previsões
predictions = model.predict(X)
print(predictions)
```



