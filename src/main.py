import time
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.ensemble import IsolationForest as IF

from src.Utility import preprocess_data, drop_outliers
from src.models.isolationForest.IsolationForest import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.Plot import anomalies_histogram_plot, auc_roc_plot, anomaly_pairplot, auc_time_plot

EXPERIMENT_FILEPATH = "./experiments results/"

def main():
    http_parameters = {
        "exp_name": "Http",
        "data_path": "../datasets/sklearn/http.csv",
        "prediction_column": "attack",
        "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    cancer_parameters = {
        "exp_name": "Cancer",
        "data_path": "../datasets/sklearn/cancer.csv",
        "prediction_column": "diagnosis",
        "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    credit_card_parameters = {
        "exp_name": "cred_card_fraud",
        "data_path": "../datasets/kaggle/credit_card_fraud/creditcard.csv",
        "prediction_column": "Class",
        "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250],
        "forest_subsampling_sizes": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [4, 8, 16, 32, 64, 128],
        "pathlength_height_limit": [4, 8, 16, 32, 64, 128],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    iris_parameters = {
        "exp_name": "Iris",
        "data_path": "../datasets/sklearn/iris.csv",
        "prediction_column": "class",
        "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    wines_parameters = {
        "exp_name": "Wines",
        "data_path": "../datasets/sklearn/wines.csv",
        "prediction_column": "class",
        "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    smtp_parameters = {
        "exp_name": "Smtp",
        "data_path": "../datasets/sklearn/smtp.csv",
        "prediction_column": "label",
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    sa_parameters = {
        "exp_name": "Sa",
        "data_path": "../datasets/sklearn/sa.csv",
        "prediction_column": "label",
        "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    gen1_parameters = {
        "exp_name": "Gen1",
        "data_path": "../datasets/sklearn/gen1.csv",
        "prediction_column": "label",
        "df_sampling_ratios": [.2, .4, .6, .8, 1],
        "df_features_ratios": [.2, .4, .6, .8, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    gen2_parameters = {
        "exp_name": "Gen2",
        "data_path": "../datasets/sklearn/gen2.csv",
        "prediction_column": "label",
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    gen3_parameters = {
        "exp_name": "Gen3",
        "data_path": "../datasets/sklearn/gen3.csv",
        "prediction_column": "label",
        "df_features_ratios": [.2, .4, .6, .8, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    landsat_parameters = {
        "exp_name": "Landsat",
        "data_path": "../datasets/uci/landsat.csv",
        "prediction_column": "label",
        "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    datasets_parameters = []
    datasets_parameters.extend([gen2_parameters, http_parameters, credit_card_parameters, cancer_parameters, landsat_parameters,
                     gen1_parameters, gen3_parameters, wines_parameters,
                     iris_parameters, sa_parameters, smtp_parameters])

    #auc_time_experiment(datasets_parameters)

    iforest_experiment_inliers(datasets_parameters)

    # for dp in datasets_parameters:
    #     run_iforest_param_experiments(dp)



def iforest_experiment_inliers(datasets_parameters):
    for dp in datasets_parameters:
        train_X, eval_X, train_y, eval_y = preprocess_data(dp["data_path"], dp["prediction_column"])
        outlier_indexes = train_y[train_y == 1].index.values
        train_X_inliers = train_X.drop(outlier_indexes)

        # training with only inliers
        iforest_experiment_data(train_X_inliers, eval_X, eval_y, f"{dp['exp_name']}_train_inliers", dp["prediction_column"])

        # training with inliers and outliers
        iforest_experiment_data(train_X, eval_X, eval_y, dp['exp_name'], dp["prediction_column"])


def auc_time_experiment(datasets_parameters):
    for dp in datasets_parameters:
        train_X, eval_X, train_y, eval_y = preprocess_data(dp["data_path"], dp["prediction_column"])

        ocsvm(train_X, eval_X, eval_y, dp["exp_name"], dp["prediction_column"])
        skl_iforest(train_X, eval_X, eval_y, dp["exp_name"], dp["prediction_column"])
        iforest_experiment_data(train_X, eval_X, eval_y, dp["exp_name"], dp["prediction_column"])

        # recombining dataframes for LOF experiment
        X = pd.concat([train_X, eval_X])
        y = pd.concat([train_y, eval_y])

        lof(X, y, dp["exp_name"], dp["prediction_column"])


def skl_iforest(train_X: DataFrame, eval_X: DataFrame, eval_y: Series, exp_name: str, pred_column: str,
                nb_tree: int = 100, subsampling_size: int = 256, df_sampling_ratio: float = 1, df_features_ratio: float = 1):

    start_train_time = time.time()
    clf = IF(n_estimators=nb_tree, max_samples=subsampling_size, warm_start=True, random_state=0)
    # training
    clf.fit(train_X)
    end_train_time = time.time()
    # evaluation
    start_eval_time = time.time()
    scores = clf.decision_function(eval_X)
    pred_y = clf.predict(eval_X)
    end_eval_time = time.time()
    pred_y[pred_y == 1] = 0
    pred_y[pred_y == -1] = 1
    auc = metrics.roc_auc_score(eval_y, pred_y)
    f1_score = metrics.f1_score(eval_y, pred_y)
    training_time = end_train_time - start_train_time
    evaluation_time = end_eval_time - start_eval_time
    total_time = training_time + evaluation_time
    print("Sklearn IForest results")
    print(f"Training time: {training_time} sec")
    print(f"Evaluation time: {evaluation_time} sec")
    print(f"Total time: {total_time}")
    print(f"AUC: {auc}")
    print(f"F1 score: {f1_score}")
    exp_parameters = {
        "experiment_name": exp_name,
        "prediction feature": pred_column,
        "pre_processing_sampling_ratio": df_sampling_ratio,
        "pre_processing_feature_ratio": df_features_ratio,
        "training_size": train_X.shape[0],
        "evaluation_size": eval_X.shape[0]
    }
    exp_results = {
        "evaluation dataset size": eval_y.shape[0],
        "AUC": auc,
        "F1_score": f1_score,
        "training_time": training_time,
        "evaluation_time": evaluation_time,
        "total_time": total_time,
    }
    iforest_infos = {
        "model name": "sklearn iforest",
        "n_estimators": clf.n_estimators,
        "contamination": clf.contamination,
        "max_samples": clf.max_samples,
        "max_features": clf.max_features,
        "bootstrap": clf.bootstrap,
        "random_state": clf.random_state,
        "verbose": clf.verbose,
        "n_jobs": clf.n_jobs,
        "warm_start": clf.warm_start
    }
    save_exp_data(f"skl_iforest_{exp_name}", exp_parameters, exp_results, model_infos=iforest_infos)
    return exp_results


def lof(X: DataFrame, y: Series, exp_name:str, pred_column: str, df_sampling_ratio: float = 1, df_features_ratio: float = 1):

    print(f"Running LOF {exp_name} experiment")

    start_time = time.time()
    print("create LOF model")
    # clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
    # clf = LocalOutlierFactor(n_neighbors=10, novelty=False)
    clf = LocalOutlierFactor(n_neighbors=20)
    print("Begin training and evaluation")
    y_pred = clf.fit_predict(X)
    # n_errors = (y_pred != ground_truth).sum()
    X_scores = clf.negative_outlier_factor_
    end_time = time.time()
    # evaluation
    # X_scores = clf.negative_outlier_factor_
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    total_time = end_time - start_time
    auc = metrics.roc_auc_score(y, y_pred)
    f1_score = metrics.f1_score(y, y_pred)
    print("LOF results")
    print(f"Total time: {total_time}")
    print(f"AUC: {auc}")
    print(f"F1 score: {f1_score}")
    exp_parameters = {
        "experiment_name": exp_name,
        "prediction_feature": pred_column,
        "pre_processing_sampling_size": df_sampling_ratio,
        "pre_processing_feature_size": df_features_ratio,
    }
    exp_results = {
        "dataset size": X.shape[0],
        "AUC": auc,
        "F1_score": f1_score,
        "total_time": total_time,
    }
    lof_infos = {
        "model_name": "LOF",
        "n_neighbors": clf.n_neighbors,
        "contamination": clf.contamination,
        "algorithm": clf.algorithm,
        "leaf_size": clf.leaf_size,
        "metric": clf.metric,
        "metric_params": clf.metric_params,
        "p": clf.p,
        "n_jobs": clf.n_jobs,
        "novelty": clf.novelty
    }
    save_exp_data(f"lof_{exp_name}", exp_parameters, exp_results, model_infos=lof_infos)
    return exp_results


def ocsvm(train_X: DataFrame, eval_X: DataFrame, eval_y: Series, exp_name: str, pred_column: str, df_sampling_ratio: float = 1, df_features_ratio: float = 1, display_boundary: bool = False):

    print(f"Running One-class SVM {exp_name} experiment")

    start_train_time = time.time()
    clf = OneClassSVM()
    # training
    print("OCSVM training")
    clf.fit(train_X)
    end_train_time = time.time()
    # evaluation
    print("OCSVM evaluation")
    start_eval_time = time.time()
    scores = clf.decision_function(eval_X)
    print("predict")
    pred_y = clf.predict(eval_X)
    end_eval_time = time.time()
    print("end predict")
    if display_boundary:
        print("display boundary")
        xx, yy = np.meshgrid(np.linspace(eval_X[:, 0].min(), eval_X[:, 0].max(), 500),
                             np.linspace(eval_X[:, 1].min(), eval_X[:, 1].max(), 500))
        print("end mesh")
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        print("decision func")
        Z = Z.reshape(xx.shape)
        print("reshape")

        print("plt")
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        plt.scatter(eval_X[:, 0], eval_X[:, 1], color='red')
        plt.xlim(eval_X[:, 0].min(), eval_X[:, 0].max())
        plt.ylim(eval_X[:, 1].min(), eval_X[:, 1].max())
        plt.title("Frontière de décision de l'OCSVM")
        plt.show()
    print(scores)
    print(pred_y)
    pred_y[pred_y == 1] = 0
    pred_y[pred_y == -1] = 1
    auc = metrics.roc_auc_score(eval_y, pred_y)
    f1_score = metrics.f1_score(eval_y, pred_y)
    training_time = end_train_time - start_train_time
    evaluation_time = end_eval_time - start_eval_time
    total_time = training_time + evaluation_time
    print("One-class SVM results")
    print(f"Training time: {training_time} sec")
    print(f"Evaluation time: {evaluation_time} sec")
    print(f"Total time: {total_time} sec")
    print(f"AUC: {auc}")
    print(f"F1 score: {f1_score}")
    exp_parameters = {
        "experiment_name": exp_name,
        "prediction_feature": pred_column,
        "pre_processing_sampling_ratio": df_sampling_ratio,
        "pre_processing_feature_ratio": df_features_ratio,
        "training_size": train_X.shape[0],
        "evaluation_size": eval_X.shape[0]
    }
    exp_results = {
        "evaluation_dataset_size": eval_y.shape[0],
        "AUC": auc,
        "F1_score": f1_score,
        "training_time": training_time,
        "evaluation_time": evaluation_time,
        "total_time": total_time,
    }
    iforest_infos = {
        "model name": "one class SVM",
        "kernel": clf.kernel,
        "degree": clf.degree,
        "nu": clf.nu,
        "gamma": clf.gamma,
        "coef0": clf.coef0,
        "tol": clf.tol,
        "shrinking": clf.shrinking,
        "verbose": clf.verbose,
        "cache_size": clf.cache_size,
        "max_iter": clf.max_iter
    }
    save_exp_data(f"ocsvm_{exp_name}", exp_parameters, exp_results, model_infos=iforest_infos)
    return exp_results


def run_iforest_param_experiments(parameters: dict):
    param_mappings = {
        "df_sampling_ratios": "df_sampling_ratio",
        "df_features_ratios": "df_features_ratio",
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
                exp_results = iforest_experiment_file(exp_name=exp_name, file_path=parameters["data_path"],
                                                      pred_column=parameters["prediction_column"],
                                                      desired_tpr=parameters.get("desired_tpr", .85),
                                                      auc_roc= parameters["auc_roc_plot"], anomaly_histogram= parameters["anomaly_histogram_plot"],
                                                      pairplot= parameters["pair_plot"], plot_save=True,
                                                      **{param_name: value})
                auc_results.append(exp_results["AUC"])
                time_results.append(exp_results["total time"])
            fig = auc_time_plot(auc_results, time_results, param_name, parameters[param_key])
            fig_exp_name = parameters["exp_name"]
            save_exp_figure(fig, EXPERIMENT_FILEPATH, f"{fig_exp_name}_auc-time_{param_name}")

def run_experiment_old(parameters: dict):
    if "df_sampling_ratios" in parameters:
        auc_results = []
        time_results = []
        for samp_ratio in parameters["df_sampling_ratios"]:
            exp_name = parameters["exp_name"] + f" sr-{samp_ratio}"
            results = iforest_experiment_file(exp_name=exp_name, file_path=parameters["data_path"],
                                              pred_column=parameters["prediction_column"], df_sampling_ratio=samp_ratio,
                                              desired_tpr=.95, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        auc_time_plot(auc_results, time_results, parameters["df_sampling_ratios"])

    if "df_feature_ratios" in parameters:
        auc_results = []
        time_results = []
        for feature_ratio in parameters["df_feature_ratios"]:
            exp_name = parameters["exp_name"] + f" fr-{feature_ratio}"
            results = iforest_experiment_file(exp_name=exp_name, file_path=parameters["data_path"],
                                              pred_column=parameters["prediction_column"], df_features_ratio=feature_ratio,
                                              desired_tpr=.85, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        auc_time_plot(auc_results, time_results, parameters["df_feature_ratios"])

    if "forest_nb_trees" in parameters:
        auc_results = []
        time_results = []
        for nb_tree in parameters["forest_nb_trees"]:
            exp_name = parameters["exp_name"] + f" nbt-{nb_tree}"
            results = iforest_experiment_file(exp_name=exp_name, file_path=parameters["data_path"],
                                              pred_column=parameters["prediction_column"], nb_tree=nb_tree, desired_tpr=.85,
                                              plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        auc_time_plot(auc_results, time_results, parameters["forest_nb_trees"])

    if "forest_subsampling_sizes" in parameters:
        auc_results = []
        time_results = []
        for subsamp_size in parameters["forest_subsampling_sizes"]:
            exp_name = parameters["exp_name"] + f" sss-{subsamp_size}"
            results = iforest_experiment_file(exp_name=exp_name, file_path=parameters["data_path"],
                                              pred_column=parameters["prediction_column"], subsampling_size=subsamp_size,
                                              desired_tpr=.85, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        auc_time_plot(auc_results, time_results, parameters["forest_subsampling_sizes"])

    if "pathlength_height_limit" in parameters:
        auc_results = []
        time_results = []
        for hl in parameters["pathlength_height_limit"]:
            exp_name = parameters["exp_name"] + f" hl-{hl}"
            results = iforest_experiment_file(exp_name=exp_name, file_path=parameters["data_path"],
                                              pred_column=parameters["prediction_column"], pathlength_height_limit=hl,
                                              desired_tpr=.85, plot_save=True)

            auc_results.append(results["AUC"])
            time_results.append(results["total time"])

        auc_time_plot(auc_results, time_results, parameters["pathlength_height_limit"])


def iforest_experiment_file(exp_name: str, file_path: str, pred_column: str, nb_tree: int = 20, subsampling_size: int = 256,
                            pathlength_height_limit: int = 0,
                            df_sampling_ratio: float = 1, df_features_ratio: float = 1, desired_tpr: float = .9,
                            auc_roc: bool = False, anomaly_histogram: bool = False, pairplot: bool = False, plot_save: bool = False) -> dict:

    train_X, eval_X, train_y, eval_y = preprocess_data(file_path, pred_column, df_sampling_ratio, df_features_ratio)

    return iforest_experiment_data(train_X, eval_X, eval_y, exp_name, pred_column,
                                   nb_tree, subsampling_size, pathlength_height_limit,
                                   df_sampling_ratio, df_features_ratio,
                                   desired_tpr,
                                   auc_roc, anomaly_histogram, pairplot, plot_save)


def iforest_experiment_data(train_X: DataFrame, eval_X: DataFrame, eval_y: Series, exp_name: str, pred_column: str,
                            nb_tree: int = 20, subsampling_size: int = 256, pathlength_height_limit: int = 0,
                            df_sampling_ratio: float = 1, df_features_ratio: float = 1,
                            desired_tpr: float = .9,
                            auc_roc: bool = False, anomaly_histogram: bool = False, pairplot: bool = False,
                            plot_save: bool = False):

    print(f"\nRunning experiment {exp_name} with the following parameters\n"
          f"- Prediction feature: {pred_column}\n"
          f"- Row sampling ratio: {df_sampling_ratio}\n"
          f"- Column sampling ratio: {df_features_ratio}\n"
          f"- Number of trees: {nb_tree}\n"
          f"- Subsampling size: {subsampling_size}\n"
          f"- Pathlength height limit: {pathlength_height_limit} nodes\n")

    print("===- Begin training phase -===")
    iforest = IsolationForest(train_X, nb_tree, subsampling_size)
    iforest.fit()

    print(f"Training time: {iforest.training_time} seconds")

    print("===- End training phase -===")
    print()
    print("===- Begin evaluation phase -===")
    scores = iforest.predict(eval_X, pathlength_height_limit)
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

    #add scores and predictions
    #eval_X['Anomaly score'] = scores
    #eval_X['Anomaly prediction'] = prediction_y

    auc_roc_fig, anomalies_hist_fig, pairplot_fig = None, None, None
    if auc_roc:
        print("Performing AUC/ROC figure")
        auc_roc_fig = auc_roc_plot(auc, eval_y, exp_name, fpr, prediction_y, tpr)
    if anomaly_histogram:
        print("Performing Anomaly histogram")
        anomalies_hist_fig = anomalies_histogram_plot(scores, eval_y, threshold)
    if pairplot:
        if eval_X.shape[1] <= 5:
            print("Performing Scatter matrix plot (pairplot)")
            pairplot_fig = anomaly_pairplot(eval_X)
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
    save_exp_data(f"iforest_{exp_name}", exp_parameters, exp_results, model_infos=iforest.get_parameters(),
                  exp_figures=exp_figures)
    plt.close()
    return exp_results


def save_exp_data(exp_title: str, exp_parameters: dict, exp_results: dict, model_infos: dict = None,
                  exp_figures: dict = None):

    new_folder_name = exp_title + " " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    folder_path = EXPERIMENT_FILEPATH + new_folder_name
    os.makedirs(folder_path)

    json_results = "experiment_data.json"
    jsonpath = os.path.join(folder_path, json_results)

    file_content = {
        "parameters": exp_parameters,
        "Model infos": model_infos,
        "Experiment results": exp_results
    }

    with open(jsonpath, "w") as outfile:
        json.dump(file_content, outfile)

    if exp_figures:
        for key, fig in exp_figures.items():
            if fig:
                save_exp_figure(fig, folder_path, key)


def save_exp_figure(figure: Figure, path: str, figure_name: str):
    figurepath = os.path.join(path, figure_name)
    figure.savefig(figurepath)


if __name__ == "__main__":
    main()
