# Isolation Forest Anomaly Detection

A Python project for studying, implementing, and comparing Isolation Forest anomaly detection. It contains a custom implementation from first principles, scikit-learn-based reference experiments, synthetic datasets, masking and swamping experiments, and plotting utilities.

The project is primarily intended as a research and educational implementation: it exposes the forest, tree, node, leaf, scoring, threshold, timing, and experimental parameters so they can be inspected and compared.

## Table of contents

- [Overview](#overview)
- [How Isolation Forest works](#how-isolation-forest-works)
- [Repository structure](#repository-structure)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Manual](#manual)
  - [Run the custom model](#run-the-custom-model)
  - [Run the full experiment suite](#run-the-full-experiment-suite)
  - [Configure an experiment](#configure-an-experiment)
  - [Preprocess CSV data](#preprocess-csv-data)
  - [Threshold and anomaly scores](#threshold-and-anomaly-scores)
- [Hyperparameters](#hyperparameters)
- [Experiment outputs](#experiment-outputs)
- [Limitations and notes](#limitations-and-notes)
- [Academic reference](#academic-reference)

## Overview

The repository provides two kinds of Isolation Forest use:

1. **Custom implementation** - `src/models/isolationForest/` builds random trees over random feature split values and computes anomaly scores from average path lengths.
2. **Experimental comparison** - `src/models/experiment/Experiment.py` compares the custom implementation with scikit-learn Isolation Forest and other optional anomaly detectors.

The experiment suite also measures:

- AUC and F1 score
- Training, evaluation, and total execution time
- The effect of the number of trees
- The effect of subsampling size
- The effect of path-length height limits
- Row and feature sampling ratios
- Training with and without outliers
- Swamping and masking behavior on synthetic clusters

## How Isolation Forest works

An Isolation Forest is an ensemble of isolation trees:

1. Random rows are sampled from the training data.
2. A random feature is selected for each node.
3. A random split value is selected between the minimum and maximum values of that feature.
4. Data points are divided into left and right children until the tree reaches its height limit or contains at most one sample.
5. The path length of a point is the number of edges required to isolate it.
6. Isolated points generally receive short path lengths and therefore high anomaly scores.

The custom implementation computes the score as:

```text
score = 2^(-mean_path_length / c(subsampling_size))
```

where `c(subsampling_size)` is the expected unsuccessful search length in a binary search tree.

An anomaly score close to `1` indicates a stronger anomaly candidate, while a score close to `0` indicates a point that is harder to isolate.

## Repository structure

```text
IsolationForest/
|-- README.md
|-- datasets/
|   `-- sklearn/
|       |-- cancer.csv
|       |-- gen1.csv
|       |-- gen2.csv
|       |-- gen3.csv
|       |-- http.csv
|       |-- iris.csv
|       |-- sa.csv
|       |-- smtp.csv
|       `-- wines.csv
|-- documents/
|   `-- Outliers detection using Isolation Forest.pdf
`-- src/
    |-- main.py
    |-- models/
    |   |-- dataset/
    |   |   |-- CreateDatasets.py
    |   |   `-- CreateSwampingMaskingCluster.py
    |   |-- experiment/
    |   |   `-- Experiment.py
    |   |-- isolationForest/
    |   |   |-- IsolationForest.py
    |   |   |-- IsolationLeaf.py
    |   |   |-- IsolationNode.py
    |   |   `-- IsolationTree.py
    |   |-- plot/
    |   |   `-- Plot.py
    |   `-- utility/
    |       `-- Utility.py
    `-- misc/
        `-- additional examples and visualizations
```

## Datasets

The bundled CSV files in `datasets/sklearn/` are already intended for the project. Labels are binary, with `1` generally representing an anomaly and `0` representing a normal observation.

| Dataset | Prediction column | Notes |
|---|---|---|
| `cancer.csv` | `diagnosis` | Breast-cancer feature dataset converted to binary anomaly labels |
| `iris.csv` | `class` | One Iris class is converted to the anomaly class |
| `wines.csv` | `class` | One Wine class is converted to the anomaly class |
| `smtp.csv` | `label` | KDD Cup 99 SMTP subset |
| `sa.csv` | `label` | KDD Cup 99 SA subset |
| `gen1.csv` | `label` | 250-feature synthetic dataset |
| `gen2.csv` | `label` | Low-dimensional synthetic dataset |
| `gen3.csv` | `label` | Small, 10-feature synthetic dataset |
| `http.csv` | `attack` | HTTP network-traffic dataset |

The default configuration also references:

- `../datasets/kaggle/credit_card_fraud/creditcard.csv`
- `../datasets/uci/landsat.csv`

These directories are not included in the repository snapshot. Add the corresponding CSV files before enabling those configurations.

`CreateDatasets.py` can generate synthetic datasets and also contains helpers for rebuilding some scikit-learn and UCI datasets.

## Requirements

The Python code uses:

- Python 3
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- `ucimlrepo` (only for UCI dataset generation)

The project has no `requirements.txt` yet, so dependencies should be installed manually.

## Installation

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn ucimlrepo
```

If PowerShell script activation is disabled, use:

```powershell
.\.venv\Scripts\activate.bat
```

or run the project through an IDE with the virtual environment selected.

## Quick start

The application entry point is `src/main.py`. Relative dataset paths are written for execution from the `src` directory:

```powershell
cd src
python main.py
```

The default entry point runs hyperparameter, inlier-removal, AUC/time, swamping, and masking experiments across the configured datasets. It can take several minutes or more because many parameter combinations are repeated ten times.

For a single custom model in Python, run from the repository root:

```python
import pandas as pd

from src.models.isolationForest.IsolationForest import IsolationForest

data = pd.DataFrame({
    "value": [10.0, 10.2, 9.8, 10.1, 10.0, 100.0],
})

model = IsolationForest(
    input_data=data,
    tree_nb=100,
    subsampling_size=6,
)
model.fit()

scores = model.predict(data)
predictions = model.predict_anomalies(scores, threshold=0.70)

print(scores)
print(predictions)
print(model.get_parameters())
```

## Manual

### Run the custom model

Create an `IsolationForest` with the following constructor arguments:

```python
IsolationForest(input_dataframe, number_of_trees, subsampling_size)
```

For example:

```python
model = IsolationForest(train_X, 100, 256)
model.fit()
scores = model.predict(eval_X)
```

The steps are:

1. Pass only numeric feature columns.
2. Call `fit()` to train the forest.
3. Call `predict(dataset)` to obtain one score per row.
4. Select a threshold and call `predict_anomalies(scores, threshold)`.

If the requested subsampling size exceeds the training row count, the implementation reduces it to the full training size.

### Run the full experiment suite

```powershell
cd src
python main.py
```

The suite uses the configurations declared in `main()`:

- Hyperparameter studies
- Training with all rows versus inliers only
- Custom versus scikit-learn Isolation Forest AUC/time comparisons
- Swamping and masking tests
- Optional AUC/ROC plots
- Optional anomaly histograms
- Optional pair plots

Because several loops use ten repetitions, editing `main.py` to reduce those values is recommended for a quick test.

### Configure an experiment

Dataset configurations are dictionaries in `src/main.py`. For example:

```python
iris_parameters = {
    "exp_name": "Iris",
    "data_path": "../datasets/sklearn/iris.csv",
    "prediction_column": "class",
    "forest_nb_trees": [3, 5, 7, 9, 13, 17],
    "forest_subsampling_sizes": generate_geometric_sequence(10, 2),
    "desired_tpr": 0.9,
    "auc_roc_plot": True,
    "anomaly_histogram_plot": True,
    "pair_plot": False,
}
```

Supported parameter keys in `run_iforest_param_experiments()` are:

| Key | Meaning |
|---|---|
| `df_sampling_ratios` | Fractions of rows to sample |
| `df_features_ratios` | Fractions of numeric columns to sample |
| `forest_nb_trees` | Numbers of trees to test |
| `forest_subsampling_sizes` | Subsample sizes to test |
| `pathlength_height_limit` | Maximum path-depth values to test |

Only parameter keys present in a dataset dictionary are executed.

### Preprocess CSV data

`preprocess_data_file()` reads a CSV and:

1. Drops rows containing missing values.
2. Keeps numeric columns.
3. Optionally performs random row or column sampling.
4. Converts a two-value string label into a binary label.
5. Performs a stratified 75/25 train/evaluation split.

```python
from src.models.utility.Utility import preprocess_data_file

train_X, eval_X, train_y, eval_y = preprocess_data_file(
    "../datasets/sklearn/cancer.csv",
    "diagnosis",
)
```

The project is sensitive to labels and feature types:

- The label column must contain at most two classes for string conversion.
- Categorical feature columns are removed.
- Missing rows are removed.
- Numeric feature values must remain comparable during tree construction.

### Threshold and anomaly scores

The custom model provides two threshold strategies:

- `tpr_threshold(y, scores, desired_tpr)` searches anomaly-score thresholds from `1.0` downward until the requested true-positive rate is reached.
- `percentile_threshold(scores, percentile_percentage)` selects a threshold at a sorted-score percentile.

Example:

```python
threshold, tpr, fpr = model.tpr_threshold(
    eval_y,
    scores,
    desired_tpr=0.90,
)
predictions = model.predict_anomalies(scores, threshold)
```

The reported score is not a probability. It is a relative isolation score, with higher values representing observations that are isolated more quickly.

## Hyperparameters

| Parameter | Default in core experiments | Effect |
|---|---:|---|
| Number of trees | 20 in `iforest_experiment_file()` | More trees usually stabilize results at higher cost |
| Subsampling size | 256 | Controls each tree's sample size and height limit |
| Tree height limit | `ceil(log2(subsampling_size))` | Limits tree growth by default |
| Path-length height limit | 0 in experiments | 0 uses the forest's computed height limit |
| Desired TPR | 0.9 | Used to select a classification threshold |

The custom implementation uses sampling **with replacement**. Splits are chosen randomly between the minimum and maximum of a selected feature.

## Experiment outputs

Experiments write JSON files and plots below the relative path `../../experiments results/` when started from `src`. In a normal clone, this may resolve outside the repository. The output is organized into directories such as:

```text
experiments results/
|-- parameters/
|-- auc_time/
`-- inliers only/
```

Each saved folder contains:

- `experiment_data.json` for parameters, model information, and metrics
- Optional AUC/ROC figures
- Optional anomaly histograms
- Optional pair plots
- Optional aggregate CSV summaries

Aggregate AUC/time comparisons include custom and scikit-learn Isolation Forest metrics. OCSVM and LOF comparison calls are present in the experiment code but disabled by default.

## Limitations and notes

- The custom implementation assumes numeric feature values and uses NumPy arrays internally.
- The label column is excluded from model features.
- Pair plots are skipped when evaluation data has more than five features.
- Random splits and random subsampling should be controlled externally if exact repeated results are required.
- The `tpr_threshold()` and experiment AUC code require evaluation labels.
- Running `src/main.py` with the credit-card or Landsat entries enabled requires missing datasets.
- Relative output and input paths depend on the working directory.
- The repository is primarily experimental and does not expose a command-line argument parser.

## Academic reference

The bundled PDF, `documents/Outliers detection using Isolation Forest.pdf`, contains the project report or study material.

The implementation follows the Isolation Forest approach described by Liu, Ting, and Zhou in *Isolation Forest* (2008). It uses random partitioning and path-length-based anomaly scoring as the central anomaly-detection mechanism.
