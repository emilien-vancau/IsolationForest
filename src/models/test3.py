import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Générer des données aléatoires avec un cluster principal
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)

# Ajouter des anomalies
outliers = np.random.uniform(low=-10, high=10, size=(6, 2))
X = np.vstack([X, outliers])

# Afficher les points
plt.scatter(X[:,0], X[:,1], c='b', s=20, label='Points normaux')
plt.scatter(outliers[:,0], outliers[:,1], c='r', s=50, marker='x', label='Anomalies')
plt.title('Cluster de points avec anomalies')
plt.legend()

fig = plt.gcf()
plt.show()

fig.savefig("normal cluster and anomalies")