import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split


def scatterplot_exemple():
    n_samples, n_outliers = 120, 40
    rng = np.random.RandomState(0)
    covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
    cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
    cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
    outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))
    X = np.concatenate([cluster_1, cluster_2, outliers])
    y = np.concatenate(
        [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    handles, labels = scatter.legend_elements()
    plt.axis("square")
    plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
    plt.title("Gaussian inliers with \nuniformly distributed outliers")
    plt.show()


def anomalies_histogram_plot(scores, y, anomaly_threshold):
    normal = scores[y == 0]
    anomalies = scores[y == 1]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

    axes[0].hist(normal, color="green", alpha=.7)
    axes[1].hist(anomalies, color="red", alpha=.7)

    axes[0].set_xlim(0, 1)
    axes[1].set_xlim(0, 1)

    # display precision of X axis
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(0.05))
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(0.05))

    axes[0].axvline(x=anomaly_threshold, color='gray', linestyle='--')
    axes[1].axvline(x=anomaly_threshold, color='gray', linestyle='--')

    axes[1].text(anomaly_threshold, 10, f'Threshold: {anomaly_threshold}', color='black')

    axes[1].set_xlabel("anomaly scores")
    axes[0].set_ylabel("normal samples counter")
    axes[1].set_ylabel("anomaly samples counter")

    fig.suptitle("Anomalies histogram")
    plt.close()

    return fig


def auc_roc_plot(auc, eval_y, exp_name, fpr, prediction_y, tpr):

    # display confusion matrix and ROC curve
    confusion_matrix = metrics.confusion_matrix(eval_y, prediction_y)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{exp_name} results")
    cm_display.plot(ax=ax1, cmap='Blues_r')
    ax2.plot(fpr, tpr, label=f"auc={auc}")
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Equality Line')
    ax2.set_xlabel("Falses Positives Rate (FPR)")
    ax2.set_ylabel("True Positives Rate (TPR)")
    ax2.legend(loc=4)
    plt.subplots_adjust(wspace=0.5)
    figure = plt.gcf()
    plt.close()

    return figure


def anomaly_pairplot(proc_df: DataFrame):
    normal_points = proc_df[proc_df["Anomaly prediction"] == 0]
    anomaly_points = proc_df[proc_df["Anomaly prediction"] == 1]

    df = pd.concat([normal_points, anomaly_points])

    pairplot = sns.pairplot(df, hue="Anomaly prediction")
    figure = pairplot.figure

    return figure


def auc_time_plot(auc_results, time_results, param_name, param_array):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.suptitle(f"AUC/TIME")

    ax1.plot(param_array, time_results, linestyle='--', color="#69b3a2", label='Temps', marker='o')
    ax2.plot(param_array, auc_results, color="#3399e6", label='AUC', marker='o')

    ax1.tick_params(axis="y", labelcolor="#69b3a2")
    ax2.tick_params(axis="y", labelcolor="#3399e6")

    time_upper_limit = max(time_results) * 1.5  # Adding 50% margin
    ax1.set_ylim(top=time_upper_limit)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    ax1.set_xlabel(f"Parameter: {param_name}")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("AUC")
    #fig.grid(True)
    plt.close()

    return fig

def auc_dist_factor_plot(auc_results, dist_factors):
    plt.figure(figsize=(10, 6))
    plt.plot(dist_factors, auc_results, marker='o', linestyle='-', color='#3399e6')

    plt.xlabel('Facteur de distance')
    plt.ylabel('AUC')
    plt.title('AUC en fonction du Facteur de distance')
    plt.legend()

    plt.grid(True)

    fig = plt.gcf()

    plt.close()

    return fig


def plot_clusters(df, label_column: str):
    plt.figure(figsize=(10, 8))
    normal_points = df[df[label_column] == 0]
    anomaly_points = df[df[label_column] == 1]

    plt.scatter(normal_points['X'], normal_points['Y'], c="blue", label=f"Normal ({len(normal_points)})")
    plt.scatter(anomaly_points['X'], anomaly_points['Y'], c="red", label=f"Anomalie ({len(anomaly_points)})")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')

    fig = plt.gcf()

    plt.close()

    return fig


def auc_contamination_hlimit_plot(contamination_levels, auc_results, hlimit_labels, colormap='viridis'):

    plt.figure(figsize=(12, 8))

    # Create a colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(auc_results)))

    for i, auc_values in enumerate(auc_results):
        plt.plot(contamination_levels, auc_values, label=f"height limit: {hlimit_labels[i]}", marker='o', color=colors[i])

    plt.xlabel('Niveau de contamination')
    plt.ylabel('AUC')
    plt.title('AUC par niveau de contamination')
    plt.legend()
    plt.grid(True)

    fig = plt.gcf()

    plt.close()

    return fig