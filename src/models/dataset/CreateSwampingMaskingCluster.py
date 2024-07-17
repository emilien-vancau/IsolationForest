import numpy as np
import matplotlib.pyplot as plt
from src.Utility import get_size_ratio

def generate_clusters(nb_points, contamination_ratio, nb_anomaly_clusters, anomalies_distance_factor, anomaly_spread=0.5):

    nb_anomalies_points = get_size_ratio(nb_points, contamination_ratio)
    nb_normal_points = nb_points - nb_anomalies_points
    points_per_anomaly_cluster = round(nb_anomalies_points / nb_anomaly_clusters)

    normal_center = np.random.uniform(-10, 10, 2)
    normal_points = np.random.randn(nb_normal_points, 2) + normal_center

    anomaly_points = []

    for _ in range(nb_anomaly_clusters):
        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)  # Normaliser pour obtenir une direction unitaire
        anomaly_center = normal_center + direction * anomalies_distance_factor * 3  # distance_factor * 3 pour échelonner la distance
        cluster_anomaly_points = anomaly_spread * np.random.randn(points_per_anomaly_cluster, 2) + anomaly_center
        anomaly_points.append(cluster_anomaly_points)

    anomaly_points = np.vstack(anomaly_points)

    # Combine anomalies and normal points
    all_points = np.vstack([normal_points, anomaly_points])

    # Création des labels
    labels = np.array([0] * nb_normal_points + [1] * len(anomaly_points))

    return all_points, labels

def plot_clusters(points, labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(points[labels == 0, 0], points[labels == 0, 1], c='blue', label='Normal')
    plt.scatter(points[labels == 1, 0], points[labels == 1, 1], c='red', label='Anomalie')
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Clusters de points normaux et d\'anomalies')
    plt.show()

def test_contamination():
    contamination_ratios = [x / 10.0 for x in range(1, 10, 1)]
    for ratio in contamination_ratios:
        points, labels = generate_clusters(nb_points=3000, contamination_ratio=ratio, nb_anomaly_clusters=3,
                                           anomalies_distance_factor=3, anomaly_spread=0.2)
        plot_clusters(points, labels)


#test_contamination()

distance_factors = [x / 10.0 for x in range(10, 50, 5)]
for factor in distance_factors:
    points, labels = generate_clusters(nb_points=3000, contamination_ratio=.1, nb_anomaly_clusters=2,
                                           anomalies_distance_factor=factor, anomaly_spread=0.2)
    plot_clusters(points, labels)

