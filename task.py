import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    confusion_matrix
)

from kneed import KneeLocator

wine = load_wine()

X = wine.data      # ознаки
y = wine.target    # мітки класів

print("Розмірність ознак:", X.shape)
print("Розмірність міток:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# визначення оптимальної кількості кластерів

inertia = []

for k in range(1, 11):

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Побудова графіка методу ліктя
plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), inertia, marker='o')

plt.xlabel("Кількість кластерів k")
plt.ylabel("Inertia")
plt.title("Метод ліктя")
plt.grid(True)

plt.show()

# визначення оптимального k
kneedle = KneeLocator(x=range(1, len(inertia)+1), y=inertia, curve="convex", direction="decreasing")
optimal_k = kneedle.elbow

print("\nОптимальна кількість кластерів:", optimal_k)


# Кластеризація методом метод k-середніх

kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto")
kmeans.fit(X_scaled)

clusters = kmeans.labels_

print("\nСправжні мітки класів:")
print(y)

print("\nМітки кластерів:")
print(clusters)

ari = adjusted_rand_score(y, clusters)
print("\nAdjusted Rand Index:", ari)

silhouette = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", silhouette)

# Матриця відповідностей
print("\nМатриця відповідностей:")
print(confusion_matrix(y, clusters))
