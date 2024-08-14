from src.models.dataset.CreateDatasets import create_anomalies_clusters_dataset
from src.models.plot.Plot import plot_clusters


def test_contamination():
    contamination_ratios = [x / 10.0 for x in range(1, 10, 1)]
    for ratio in contamination_ratios:
        df = create_anomalies_clusters_dataset(nb_points=3000, contamination_ratio=ratio, nb_anomaly_clusters=1,
                                               anomalies_distance_factor=1.5, anomaly_spread=0.6)

        plot_clusters(df, "labels")


test_contamination()

def test_distance_factor():
    global distance_factors, aucs
    distance_factors = [x / 10.0 for x in range(10, 50, 5)]
    for factor in distance_factors:
        points, labels = create_anomalies_clusters_dataset(nb_points=3000, contamination_ratio=.1, nb_anomaly_clusters=1,
                                                           anomalies_distance_factor=factor, anomaly_spread=0.4)




