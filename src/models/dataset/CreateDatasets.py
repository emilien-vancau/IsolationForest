from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_kddcup99, make_classification
from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

from src.Utility import get_size_ratio


def create_anomaly_dataset(n_samples: int, n_features: int, outiers_percentage: float, n_informative: int,
                           n_redundant: int,
                           class_sep: int, n_clusters_per_class: int, random_state: int = 4):

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=n_redundant,
                               n_clusters_per_class=n_clusters_per_class,
                               n_informative=n_informative, class_sep=class_sep, weights=[1], flip_y=0,
                               random_state=random_state)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["label"] = y

    n_outliers = get_size_ratio(n_samples, outiers_percentage)

    # randomly select inliers as outliers
    outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)
    outliers = np.random.uniform(low=df.min().min(), high=df.max().max(), size=(n_outliers, n_features))

    # replace randomly selected inliers with outliers
    df.loc[outlier_indices, df.columns[:-1]] = outliers
    df.loc[outlier_indices, 'label'] = 1

    # shuffle dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

def create_anomaly_dataset_experimental(n_samples: int, n_features: int, outiers_percentage: float, n_informative: int,
                           n_redundant: int,
                           class_sep: int, n_clusters_per_class: int, random_state: int = 4):

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=n_redundant,
                               n_clusters_per_class=n_clusters_per_class,
                               n_informative=n_informative, class_sep=class_sep, weights=[1], flip_y=0,
                               random_state=random_state)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["label"] = y

    n_outliers = get_size_ratio(n_samples, outiers_percentage)

    # randomly select inliers as outliers
    outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)
    #outliers = np.random.uniform(low=df.min().min(), high=df.max().max(), size=(n_outliers, n_features))

    # generate anomalies close to inliers
    anomaly_magnitude = 0.5  # adjust this value to control the anomaly distance
    outliers = df.loc[outlier_indices, df.columns[:-1]] + np.random.normal(loc=0, scale=anomaly_magnitude,
                                                                           size=(n_outliers, n_features))

    # replace randomly selected inliers with outliers
    df.loc[outlier_indices, df.columns[:-1]] = outliers
    df.loc[outlier_indices, 'label'] = 1

    # shuffle dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def create_anomalies_clusters_dataset(nb_points, contamination_ratio, nb_anomaly_clusters, anomalies_distance_factor, anomaly_spread=0.5):

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

    df = pd.DataFrame(data=all_points, columns=['X', 'Y'])
    df["labels"] = labels

    return df


def make_datasets():
    # Get Breast cancer dataset
    cancer = load_breast_cancer()
    df_cancer = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df_cancer["diagnosis"] = cancer.target
    df_cancer["diagnosis"] = df_cancer['diagnosis'].apply(lambda x: 1 if x == 0 else 0)
    cancer_proportions = df_cancer['diagnosis'].value_counts(normalize=True)
    print("Cancer Proportions")
    print(cancer_proportions)

    # Get Iris dataset
    iris = load_iris()
    df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_iris["class"] = iris.target
    df_iris["class"] = df_iris["class"].apply(lambda x: 1 if x == 2 else 0)
    iris_proportions = df_iris['class'].value_counts(normalize=True)
    print("Iris Proportions")
    print(iris_proportions)

    # Get Wine dataset
    wine = load_wine()
    df_wines = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df_wines["class"] = wine.target
    df_wines["class"] = df_wines["class"].apply(lambda x: 1 if x == 1 else 0)
    wine_proportions = df_wines['class'].value_counts(normalize=True)
    print("Wine Proportions")
    print(wine_proportions)

    # Get SMTP dataset
    kddcup99_smtp = fetch_kddcup99(subset="smtp", random_state=42)
    df_smtp = pd.DataFrame(data=kddcup99_smtp.data, columns=kddcup99_smtp.feature_names)
    df_smtp["label"] = kddcup99_smtp.target
    # print(df_smtp.loc[df_smtp['label'] != b'normal.'])
    df_smtp["label"] = df_smtp["label"].apply(lambda x: 0 if x == b'normal.' else 1)
    # val_uni_smtp = df_smtp["attack"].value_counts()
    smtp_proportions = df_smtp['label'].value_counts(normalize=True)
    print("SMTP Proportions")
    print(smtp_proportions)

    # Get SA dataset
    kddcup99_sa = fetch_kddcup99(subset="SA", random_state=42)
    df_sa = pd.DataFrame(data=kddcup99_sa.data, columns=kddcup99_sa.feature_names)
    df_sa["label"] = kddcup99_sa.target
    df_sa["label"] = df_sa["label"].apply(lambda x: 0 if x == b'normal.' else 1)
    # val_uni_sa = df_sa["attack"].value_counts()
    sa_proportions = df_sa['label'].value_counts(normalize=True)
    print("SA Proportions")
    print(sa_proportions)

    # Generate classification datasets
    df_gen1 = create_anomaly_dataset_experimental(n_samples=1000, n_features=250, outiers_percentage=.07,
                                     n_informative=1, n_redundant=0, class_sep=1, n_clusters_per_class=1)
    gen1_proportions = df_gen1['label'].value_counts(normalize=True)
    print("Gen1 Proportions")
    print(gen1_proportions)


    df_gen2 = create_anomaly_dataset_experimental(n_samples=300, n_features=3, outiers_percentage=.01,
                                     n_informative=1, n_redundant=0, class_sep=1, n_clusters_per_class=1)
    gen2_proportions = df_gen2['label'].value_counts(normalize=True)
    print("Gen2 Proportions")
    print(gen2_proportions)

    df_gen3 = create_anomaly_dataset_experimental(n_samples=30, n_features=10, outiers_percentage=.15,
                                     n_informative=2, n_redundant=0, class_sep=1, n_clusters_per_class=2)
    gen3_proportions = df_gen3['label'].value_counts(normalize=True)
    print("Gen3 Proportions")
    print(gen3_proportions)

    # landsat
    statlog_landsat_satellite = fetch_ucirepo(id=146)

    df_landsat = statlog_landsat_satellite.data.features
    df_landsat["label"] = statlog_landsat_satellite.data.targets
    df_landsat["label"] = df_landsat["label"].apply(lambda x: 1 if x == 4 else 0)

    landsat_proportions = df_landsat['label'].value_counts(normalize=True)
    print("X Proportions")
    print(landsat_proportions)

    #df_cancer.to_csv("cancer.csv", index=False)
    #df_iris.to_csv("iris.csv", index=False)
    #df_wines.to_csv("wines.csv", index=False)
    #df_smtp.to_csv("smtp.csv", index=False)
    #df_sa.to_csv("sa.csv", index=False)
    #df_gen1.to_csv("gen1.csv", index=False)
    df_gen2.to_csv("gen2.csv", index=False)
    df_gen3.to_csv("gen3.csv", index=False)
    #df_landsat.to_csv("landsat.csv", index=False)



make_datasets()





