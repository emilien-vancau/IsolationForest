import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

large_cluster_mean = [0, 0]
large_cluster_cov = [[1, 0], [0, 1]]  # Covariance matrice (identité)
large_cluster_points = np.random.multivariate_normal(large_cluster_mean, large_cluster_cov, 500) #changer ici pour plus de points du grand cluster

# Générer plusieurs petits clusters d'anomalies autour du grand cluster
anomaly_clusters_means = [[3, 3], [-3, -3], [5, -4], [1, 4]] #nombres de clusters d'anomalies et leur position par rapport au centre
anomaly_clusters_points = []
for mean in anomaly_clusters_means:
    points = np.random.multivariate_normal(mean, [[0.1, 0], [0, 0.1]], 50)  #changer ici pour plus de points des clusters d'anomalies
    anomaly_clusters_points.append(points)


anomaly_clusters_points = np.vstack(anomaly_clusters_points)

plt.figure(figsize=(10, 10))
plt.scatter(large_cluster_points[:, 0], large_cluster_points[:, 1], c='blue', label='Inliers')
plt.scatter(anomaly_clusters_points[:, 0], anomaly_clusters_points[:, 1], c='red', label='Outliers')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Masking and Swamping effects')
plt.grid(True)
plt.show()
