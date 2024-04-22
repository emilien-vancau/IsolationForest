import math
import time
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.ensemble import IsolationForest as IF
from src.models.isolationForest.IsolationForest import IsolationForest


def main():
    parameters = {
        "data_path": "../datasets/others/http.csv",
        "prediction_column": "attack",
        "df_sampling_sizes": [.2, .4, .6, .8, 1],
        "df_feature_sizes": [.2, .4, .6, .8, 1],
        "forest_nb_trees": [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250],
        "forest_subsampling_sizes": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "pathlength_height_limit": [4, 8, 16, 32, 64, 128]
    }

    run_experiment(exp_name="Http", file_path=parameters["data_path"], pred_column=parameters["prediction_column"],
                   tree_nb=20, subsampling_size=256, pathlength_height_limit=10, df_sampling_size=100000,
                   desired_tpr=.85,
                   display_plot=True, is_plot_save=True)

    # dataset preprocess
    # #train_X, eval_X, _, _ = preprocess_data("../datasets/kaggle/financal_Data/BBCA.JK_weekly_new.csv", None)
    # train_X, eval_X, train_y, eval_y = preprocess_data("../datasets/others/http.csv", "attack", 100000)
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

    print("-------------------------------------------------------------------")

    # clf = IF(n_estimators=300, max_samples=256, warm_start=True, random_state=0)
    # clf.fit(train_X)
    # scores2 = clf.decision_function(eval_x)
    # print(scores2)


def run_experiment(exp_name: str, file_path: str, pred_column: str, tree_nb: int = 20, subsampling_size: int = 256, pathlength_height_limit: int = None,
                   df_sampling_size: int = None, df_features_size: int = None, desired_tpr: float = None,
                   display_plot: bool = False, is_plot_save: bool = False):

    train_X, eval_X, train_y, eval_y = __preprocess_data(file_path, pred_column, df_sampling_size, df_features_size)

    print("===- Begin training phase -===")
    iforest = IsolationForest(train_X, tree_nb, subsampling_size)
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
    print(f"AUC: {auc}")

    exp_parameters = {
        "prediction feature": pred_column,
        "sampling size": df_sampling_size,
        "feature size": df_features_size,
        "trees number": tree_nb,
        "subsampling size": subsampling_size,
        "pathlength height limit": pathlength_height_limit
    }

    exp_results = {
        "AUC": auc,
        "TPR": tpr[1],
        "FPR": fpr[1],
        "anomaly threshold": threshold,
        "total time": iforest.training_time + iforest.evaluation_time,
    }

    figure = __create_plot(auc, eval_y, exp_name, fpr, prediction_y, tpr, is_plot_save)

    if display_plot:
        plt.show()

    __save_exp_data(exp_name, exp_parameters, iforest.get_parameters(), exp_results, figure)


def __create_plot(auc, eval_y, exp_name, fpr, prediction_y, tpr, is_plot_save):
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

    if is_plot_save:
        return figure

    return


def __preprocess_data(file_path: str, prediction_column: str, sample_row_size=None, sample_column_size=None):
    df = pd.read_csv(file_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    # drop column which contains categorical values and drop row with missing values
    proc_df = df[df.select_dtypes(include=['float64', 'int64']).columns.tolist()].dropna()

    # randomly pre-sample dataset
    if sample_row_size:
        if proc_df.shape[0] >= sample_row_size:
            proc_df = proc_df.sample(sample_row_size)
        else:
            print("sample size too high for dataframe")
            return

    if sample_column_size:
        if sample_column_size <= proc_df.shape[1]:
            proc_df = proc_df.sample(n=sample_column_size, axis='columns')
        else:
            print("not enough columns to do random features selection")
            return

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
                    exp_figure: Figure = None):
    new_folder_name = exp_title + " " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    folder_path = "./experiments results/" + new_folder_name
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

    if exp_figure:
        figure_results = "experiment_figure.png"
        figurepath = os.path.join(folder_path, figure_results)
        exp_figure.savefig(figurepath)


if __name__ == "__main__":
    main()
