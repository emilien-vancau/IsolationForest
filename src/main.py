import math
import time
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from sklearn.ensemble import IsolationForest as IF
from src.models.isolationForest.IsolationForest import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


EXPERIMENT_FILEPATH = "./experiments results/"

def main():
    http_parameters = {
        "exp_name": "Http",
        "data_path": "../datasets/others/http.csv",
        "prediction_column": "attack",
        "df_sampling_ratios": [.2, .4, .6, .8, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250],
        "forest_subsampling_sizes": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [4, 8, 16, 32, 64, 128],
        "pathlength_height_limit": [4, 8, 16, 32, 64, 128],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    cancer_parameters = {
        "exp_name": "Cancer",
        "data_path": "../datasets/others/cancer.csv",
        "prediction_column": "diagnosis",
        #"df_sampling_ratios": [.2, .4, .6, .8, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250],
        #"forest_subsampling_sizes": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        #"tree_height_limit": [4, 8, 16, 32, 64, 128],
        #"pathlength_height_limit": [4, 8, 16, 32, 64, 128],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": True
    }

    #run_experiment(http_parameters)
    #run_iforest_experiment(cancer_parameters)

    # parameters = {
    #     "exp_name": "Ddos",
    #     "data_path": "../datasets/kaggle/ddos/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    #     "prediction_column": "Label",
    #     "df_drop_columns": [""],
    #     "df_sampling_ratios": [.2, .4, .6, .8, 1],
    # }



    # iforest_experiment(exp_name=parameters["exp_name"], file_path=parameters["data_path"], pred_column=parameters["prediction_column"],
    #                    nb_tree=20, subsampling_size=256, pathlength_height_limit=10, df_sampling_ratio=.2,
    #                    desired_tpr=.85,
    #                    plot_save=True)

    # dataset preprocess
    # #train_X, eval_X, _, _ = preprocess_data("../datasets/kaggle/financal_Data/BBCA.JK_weekly_new.csv", None)
    # train_X, eval_X, train_y, eval_y = preprocess_data("../datasets/others/http.csv", "attack", .2)
    # #train_X, eval_X, train_y, eval_y = preprocess_data("../datasets/kaggle/ddos/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Label")
    #
    # # print(list(financal_data.columns.values)) #get columns name
    # # print(financal_data['Date'].tolist()) #get all values from a column
    # # print(X_train_set)
    # # print(X_train_set.index.values)

    # # add new columns with anomaly score and prediction for each observations
    # # eval_X['anomaly score'], eval_y['anomaly score'] = scores, scores
    # # eval_X['anomaly prediction'], eval_y['anomaly prediction'] = prediction_y, prediction_y
    # # print(eval_X.sort_values(by='anomaly score', ascending=False))
    # # print(eval_y.sort_values(by='anomaly score', ascending=False))
    #

    #scatter_exemple()

    print("-------------------------------------------------------------------")

    sklearn_iforest(http_parameters["data_path"], http_parameters["prediction_column"])
    #lof_experiment(http_parameters["data_path"], http_parameters["prediction_column"])


def scatter_exemple():
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


def sklearn_iforest(file_path: str, pred_column: str, nb_tree: int = 20, subsampling_size: int = 256, df_sampling_ratio: float = None, df_features_ratio: float = None):

    train_X, eval_X, train_y, eval_y = preprocess_data(file_path, pred_column, df_sampling_ratio, df_features_ratio)

    start_train_time = time.time()
    clf = IF(n_estimators=nb_tree, max_samples=subsampling_size, warm_start=True, random_state=0)

    #training
    clf.fit(train_X)
    end_train_time = time.time()

    #evaluation
    start_eval_time = time.time()
    scores = clf.decision_function(eval_X)
    pred_y = clf.predict(eval_X)
    end_eval_time = time.time()

    pred_y[pred_y == 1] = 0
    pred_y[pred_y == -1] = 1

    auc = metrics.roc_auc_score(eval_y, pred_y)
    f1_score = metrics.f1_score(eval_y, pred_y)

    print(f"AUC: {auc}")
    print(f"F1 score: {f1_score}")


def lof_experiment(file_path: str, pred_column: str, n_neighbors: int = 20, df_sampling_ratio: float = None, df_features_ratio: float = None):

    train_X, eval_X, train_y, eval_y = preprocess_data(file_path, pred_column, df_sampling_ratio, df_features_ratio)

    #clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)

    #training
    pred = clf.fit_predict(train_X)

    #evaluation
    #X_scores = clf.negative_outlier_factor_

    pred[pred == 1] = 0
    pred[pred == -1] = 1

    auc = metrics.roc_auc_score(train_y, pred)
    f1_score = metrics.f1_score(train_y, pred)

    print(f"AUC: {auc}")
    print(f"F1 score: {f1_score}")

    #print(X_scores)


def svm_experiment(file_path: str, pred_column: str, kernel: str, nu: float, df_sampling_ratio: float = None, df_features_ratio: float = None):
    pass


def run_iforest_experiment(parameters: dict):
    param_mappings = {
        "df_sampling_ratios": "df_sampling_ratio",
        "df_feature_ratios": "df_features_ratio",
        "forest_nb_trees": "nb_tree",
        "forest_subsampling_sizes": "subsampling_size",
        "pathlength_height_limit": "pathlength_height_limit",
    }

    for param_key, param_name in param_mappings.items():
        if param_key in parameters:
            auc_results = []
            time_results = []
            for value in parameters[param_key]:
                exp_name = parameters["exp_name"] + f" {param_name}-{value}"
                exp_results = iforest_experiment(exp_name=exp_name, file_path=parameters["data_path"],
                                            pred_column=parameters["prediction_column"],
                                            desired_tpr=parameters.get("desired_tpr", .85),
                                            auc_roc= parameters["auc_roc_plot"], anomaly_histogram= parameters["anomaly_histogram_plot"],
                                            pairplot= parameters["pair_plot"], plot_save=True,
                                            **{param_name: value})
                auc_results.append(exp_results["AUC"])
                time_results.append(exp_results["total time"])
            fig = __auc_time_plot(auc_results, time_results, param_name, parameters[param_key])
            fig_exp_name = parameters["exp_name"]
            __save_exp_figure(fig, EXPERIMENT_FILEPATH, f"{fig_exp_name}_auc-time_{param_name}")

def run_experiment_old(parameters: dict):
    if "df_sampling_ratios" in parameters:
        auc_results = []
        time_results = []
        for samp_ratio in parameters["df_sampling_ratios"]:
            exp_name = parameters["exp_name"] + f" sr-{samp_ratio}"
            results = iforest_experiment(exp_name=exp_name, file_path=parameters["data_path"],
                                         pred_column=parameters["prediction_column"], df_sampling_ratio=samp_ratio,
                                         desired_tpr=.95, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        __auc_time_plot(auc_results, time_results, parameters["df_sampling_ratios"])

    if "df_feature_ratios" in parameters:
        auc_results = []
        time_results = []
        for feature_ratio in parameters["df_feature_ratios"]:
            exp_name = parameters["exp_name"] + f" fr-{feature_ratio}"
            results = iforest_experiment(exp_name=exp_name, file_path=parameters["data_path"],
                                         pred_column=parameters["prediction_column"], df_features_ratio=feature_ratio,
                                         desired_tpr=.85, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        __auc_time_plot(auc_results, time_results, parameters["df_feature_ratios"])

    if "forest_nb_trees" in parameters:
        auc_results = []
        time_results = []
        for nb_tree in parameters["forest_nb_trees"]:
            exp_name = parameters["exp_name"] + f" nbt-{nb_tree}"
            results = iforest_experiment(exp_name=exp_name, file_path=parameters["data_path"],
                                         pred_column=parameters["prediction_column"], nb_tree=nb_tree, desired_tpr=.85,
                                         plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        __auc_time_plot(auc_results, time_results, parameters["forest_nb_trees"])

    if "forest_subsampling_sizes" in parameters:
        auc_results = []
        time_results = []
        for subsamp_size in parameters["forest_subsampling_sizes"]:
            exp_name = parameters["exp_name"] + f" sss-{subsamp_size}"
            results = iforest_experiment(exp_name=exp_name, file_path=parameters["data_path"],
                                         pred_column=parameters["prediction_column"], subsampling_size=subsamp_size,
                                         desired_tpr=.85, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        __auc_time_plot(auc_results, time_results, parameters["forest_subsampling_sizes"])

    if "pathlength_height_limit" in parameters:
        auc_results = []
        time_results = []
        for hl in parameters["pathlength_height_limit"]:
            exp_name = parameters["exp_name"] + f" hl-{hl}"
            results = iforest_experiment(exp_name=exp_name, file_path=parameters["data_path"],
                                         pred_column=parameters["prediction_column"], pathlength_height_limit=hl,
                                         desired_tpr=.85, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        __auc_time_plot(auc_results, time_results, parameters["pathlength_height_limit"])


def iforest_experiment(exp_name: str, file_path: str, pred_column: str, nb_tree: int = 20, subsampling_size: int = 256,
                        pathlength_height_limit: int = 0,
                        df_sampling_ratio: float = None, df_features_ratio: float = None, desired_tpr: float = None,
                        auc_roc: bool = False, anomaly_histogram: bool = False, pairplot: bool = False, plot_save: bool = False) -> dict:

    print(f"\nRunning experiment {exp_name} with the following parameters\n"
          f"- Prediction feature: {pred_column}\n"
          f"- Row sampling ratio: {df_sampling_ratio}\n"
          f"- Column sampling ratio: {df_features_ratio}\n"
          f"- Number of trees: {nb_tree}\n"
          f"- Subsampling size: {subsampling_size}\n"
          f"- Pathlength height limit: {pathlength_height_limit} nodes\n")

    train_X, eval_X, train_y, eval_y = preprocess_data(file_path, pred_column, df_sampling_ratio, df_features_ratio)

    print("===- Begin training phase -===")
    iforest = IsolationForest(train_X, nb_tree, subsampling_size)
    iforest.fit()
    print(f"Training time: {iforest.training_time} seconds")
    print("===- End training phase -===")
    print()

    print("===- Begin evaluation phase -===")
    scores = iforest.score(eval_X, pathlength_height_limit)
    print(f"Score compute in: {iforest.evaluation_time} seconds")
    print("===- End evaluation phase -===")
    print()

    # threshold and prediction
    threshold, _, _ = iforest.tpr_threshold(eval_y, scores, desired_tpr)
    prediction_y = iforest.predict_anomalies(scores, threshold)
    print(f"Threshold at: {threshold}")

    # ROC and AUC computation
    fpr, tpr, _ = metrics.roc_curve(eval_y, prediction_y)
    auc = metrics.roc_auc_score(eval_y, prediction_y)
    f1_score = metrics.f1_score(eval_y, prediction_y)

    print(f"AUC: {auc}")
    print(f"F1 score: {f1_score}")
    print()

    eval_X['Anomaly score'] = scores
    eval_X['Anomaly prediction'] = prediction_y

    auc_roc_fig, anomalies_hist_fig, pairplot_fig = None, None, None
    if auc_roc:
        print("Performing AUC/ROC figure")
        auc_roc_fig = __auc_roc_plot(auc, eval_y, exp_name, fpr, prediction_y, tpr)
    if anomaly_histogram:
        print("Performing Anomaly histogram")
        anomalies_hist_fig = __anomalies_histogram_plot(scores, eval_y, threshold)
    if pairplot:
        if eval_X.shape[1] <=5:
            print("Performing Scatter matrix plot (pairplot)")
            pairplot_fig = __anomaly_pairplot(eval_X)
        else:
            print("unable to perform scatter matrix because of the high number of features")

    exp_parameters = {
        "prediction feature": pred_column,
        "pre-processing sampling size": df_sampling_ratio,
        "pre-processing feature size": df_features_ratio,
        "trees number": nb_tree,
        "subsampling size": subsampling_size,
        "pathlength height limit": pathlength_height_limit
    }

    exp_results = {
        "evaluation dataset size": eval_y.shape[0],
        "AUC": auc,
        "F1 score": f1_score,
        "TPR": tpr[1],
        "FPR": fpr[1],
        "anomaly threshold": threshold,
        "total time": iforest.training_time + iforest.evaluation_time,
    }

    exp_figures = {}
    if plot_save:
        exp_figures = {
            "AUC-ROC figures": auc_roc_fig,
            "Anomalies histogram": anomalies_hist_fig,
            "Scatter matrix": pairplot_fig
        }

    __save_exp_data(exp_name, exp_parameters, iforest.get_parameters(), exp_results, exp_figures)

    plt.close()

    return exp_results


def __anomalies_histogram_plot(scores, y, anomaly_threshold):
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

    return fig


def __auc_roc_plot(auc, eval_y, exp_name, fpr, prediction_y, tpr):

    # display confusion matrix and ROC curve
    confusion_matrix = metrics.confusion_matrix(eval_y, prediction_y)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{exp_name} results")
    cm_display.plot(ax=ax1, cmap='Blues_r')
    ax2.plot(fpr, tpr, label=f"auc={auc}")
    ax2.set_xlabel("Falses Positives Rate (FPR)")
    ax2.set_ylabel("True Positives Rate (TPR)")
    ax2.legend(loc=4)
    plt.subplots_adjust(wspace=0.5)
    figure = plt.gcf()

    return figure


def __anomaly_pairplot(proc_df: DataFrame):
    normal_points = proc_df[proc_df["Anomaly prediction"] == 0]
    anomaly_points = proc_df[proc_df["Anomaly prediction"] == 1]

    df = pd.concat([normal_points, anomaly_points])

    pairplot = sns.pairplot(df, hue="Anomaly prediction")
    figure = pairplot.figure

    return figure


def __auc_time_plot(auc_results, time_results, param_name, param_array):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.suptitle(f"AUC/TIME title")

    ax1.plot(param_array, time_results, color="#69b3a2", label='Temps', marker='o')
    ax2.plot(param_array, auc_results, color="#3399e6", label='AUC', marker='o')

    ax1.tick_params(axis="y", labelcolor="#69b3a2")
    ax2.tick_params(axis="y", labelcolor="#3399e6")

    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    ax1.set_xlabel(f"Parameter: {param_name}")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("AUC")
    #fig.grid(True)

    return fig


def get_size_ratio(size: int, ratio: float):
    if 0 < ratio <= 1:
        return round(size * ratio)
    print("ratio should be between 0 (exclude) and 1")
    return


def preprocess_data(file_path: str, prediction_column: str, sample_row_ratio=None, sample_column_ratio=None):
    df = pd.read_csv(file_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    # drop column which contains categorical values and drop row with missing values
    proc_df = df[df.select_dtypes(include=['float64', 'int64']).columns.tolist()].dropna()

    # randomly pre-sample dataset
    if sample_row_ratio:
        sample_row = get_size_ratio(proc_df.shape[0], sample_row_ratio)
        proc_df = proc_df.sample(sample_row)

    if sample_column_ratio:
        sample_column = get_size_ratio(proc_df.shape[1], sample_column_ratio)
        if sample_column < 2:
            sample_column = 2
            print("not enough columns to perform isolation processing, sample column set to 2")
        proc_df = proc_df.sample(sample_column, axis="columns")

    train_X, eval_X, train_y, eval_y = None, None, None, None

    if prediction_column:

        # add prediction column to processed dataframe
        proc_df[prediction_column] = df[prediction_column]

        # check if prediction data are strings and if it's only 2 possible value that can be used as a boolean
        if proc_df[prediction_column].dtype == 'object':
            if len(proc_df[prediction_column].unique()) == 2:
                # perform one hot encoding to convert values to boolean value
                proc_df[prediction_column] = pd.get_dummies(proc_df[prediction_column], drop_first=True).sum(axis=1)
            else:
                print("unable to convert anomaly labels to booleans values")
                return

        X, y = proc_df.drop(prediction_column, axis=1), proc_df[prediction_column]
        # split input data
        train_X, eval_X, train_y, eval_y = train_test_split(X, y, random_state=1)
        return train_X, eval_X, train_y, eval_y

    else:
        # split input data
        train_X, eval_X = train_test_split(proc_df, random_state=1)
        return train_X, eval_X, train_y, eval_y


def compute_auc(anomalies_labels, anomalies_scores):
    df = pd.DataFrame({'score': anomalies_scores, 'label': anomalies_labels})
    true_anomalies = df[df['label'] == 1]
    df.sort_values(by='score', ascending=False)

    true_anomalies_cnt = true_anomalies.shape[0]
    true_normal_points_cnt = df.shape[0] - true_anomalies_cnt

    # get n firsts row in the ranked list
    df = df.head(true_anomalies_cnt)

    S = sum(df.index.values)

    # S = sum(rank for rank, label in enumerate(scores_labels, start=1) if label)

    AUC = (S - ((true_anomalies_cnt ** 2 + true_anomalies_cnt) / 2)) / (true_anomalies_cnt * true_normal_points_cnt)

    return AUC


def __save_exp_data(exp_title: str, exp_parameters: dict, forest_infos: dict, exp_results: dict,
                    exp_figures: dict = None):

    new_folder_name = exp_title + " " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    folder_path = EXPERIMENT_FILEPATH + new_folder_name
    os.makedirs(folder_path)

    json_results = "experiment_data.json"
    jsonpath = os.path.join(folder_path, json_results)

    file_content = {
        "parameters": exp_parameters,
        "Forest infos": forest_infos,
        "Experiment results": exp_results
    }

    with open(jsonpath, "w") as outfile:
        json.dump(file_content, outfile)

    if exp_figures:
        for key, fig in exp_figures.items():
            if fig:
                __save_exp_figure(fig, folder_path, key)


def __save_exp_figure(figure: Figure, path: str, figure_name: str):
    figurepath = os.path.join(path, figure_name)
    figure.savefig(figurepath)


if __name__ == "__main__":
    main()
