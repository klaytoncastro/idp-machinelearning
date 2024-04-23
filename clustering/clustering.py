import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Carregando o dataset
data = load_wine()
X = data.data
y = data.target

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando o K-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Reduzindo a dimensionalidade para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotando os resultados
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.title('Clusterização do Dataset de Vinhos com K-means')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(scatter)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Carregando o dataset
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando o K-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Reduzindo a dimensionalidade para visualização com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotando os resultados do clustering
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k', s=50)
plt.title('Clusterização do Dataset de Vinhos com K-means')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(scatter)
plt.show()

# Visualizando as cargas dos componentes principais
loadings = pca.components_.T
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(range(len(feature_names)), loadings[:, 0], label='PC1')
ax.bar(range(len(feature_names)), loadings[:, 1], bottom=loadings[:, 0], label='PC2')
plt.xticks(range(len(feature_names)), feature_names, rotation=90)
plt.ylabel('Cargas dos Componentes Principais')
plt.title('Contribuição das Variáveis aos Componentes Principais')
plt.legend()
plt.tight_layout()
plt.show()