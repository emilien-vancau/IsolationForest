from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_kddcup99, make_classification
import pandas as pd
import numpy as np

# Get Breast cancer dataset
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df_cancer["diagnosis"] = cancer.target
df_cancer["diagnosis"] = df_cancer['diagnosis'].apply(lambda x: 1 if x == 0 else 0)

# Get Iris dataset
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris["class"] = iris.target
df_iris["class"] = df_iris["class"].apply(lambda x: 1 if x == 2 else 0)

# Get Wine dataset
wine = load_wine()
df_wines = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df_wines["class"] = wine.target
df_wines["class"] = df_wines["class"].apply(lambda x: 1 if x == 1 else 0)

# Get SMTP dataset
kddcup99_smtp = fetch_kddcup99(subset="smtp", random_state=42)
df_smtp = pd.DataFrame(data=kddcup99_smtp.data, columns=kddcup99_smtp.feature_names)
df_smtp["label"] = kddcup99_smtp.target
df_smtp["label"] = df_smtp["label"].apply(lambda x: 1 if x != b'normal' else 0)
#val_uni_smtp = df_smtp["attack"].value_counts()

# Get SA dataset
kddcup99_sa = fetch_kddcup99(subset="SA", random_state=42)
df_sa = pd.DataFrame(data=kddcup99_sa.data, columns=kddcup99_sa.feature_names)
df_sa["label"] = kddcup99_sa.target
df_sa["label"] = df_sa["label"].apply(lambda x: 1 if x != b'normal' else 0)
#val_uni_sa = df_sa["attack"].value_counts()

# Generate classification datasets
X, y = make_classification(n_samples=1000, n_features=10, n_redundant=0, n_clusters_per_class=1, weights=[.7], flip_y=0, random_state=4)
#unique, counts = np.unique(y, return_counts=True)
df_gen1 = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df_gen1["label"] = y


X, y = make_classification(n_samples=300, n_features=3, n_redundant=0, n_clusters_per_class=1, weights=[.999], flip_y=0, random_state=4)
df_gen2 = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df_gen2["label"] = y

X, y = make_classification(n_samples=30, n_features=10, n_redundant=0, n_clusters_per_class=1, weights=[.85], flip_y=0, random_state=4)
df_gen3 = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df_gen3["label"] = y

df_cancer.to_csv("cancer.csv", index=False)
df_iris.to_csv("iris.csv", index=False)
df_wines.to_csv("wines.csv", index=False)
df_smtp.to_csv("smtp.csv", index=False)
df_sa.to_csv("sa.csv", index=False)
df_gen1.to_csv("gen1.csv", index=False)
df_gen2.to_csv("gen2.csv", index=False)
df_gen3.to_csv("gen3.csv", index=False)

