from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Données sur les température et vente de glaces
data = np.array([
    [15.4, 323],
    [14.9, 175],
    [13.2, 345],
    [18.5, 420],
    [21.1, 512],
    [13.4, 484],
    [22.2, 600],
    [28.7, 522],
    [16.8, 433],
    [23.7, 424],
    [18.3, 423]
])

# Application de la méthode KMeans pour répartir les données en 2 groupes
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# Couleurs des clusters
colors = ['purple' if label == 0 else 'orange' for label in kmeans.labels_]

# Représentation graphique
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=colors, marker='o', edgecolor='k')
plt.title('Répartition des ventes de glaces en fonction de la température')
plt.xlabel('Température')
plt.ylabel('Vente de glaces')
plt.grid(True)

# Affichage des centres des clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='+', s=200)

plt.show()

print("Centres des clusters :", kmeans.cluster_centers_)
