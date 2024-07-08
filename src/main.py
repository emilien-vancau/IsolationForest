import logging
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

from src.Utility import preprocess_data
from src.models.isolationForest.IsolationForest import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.Plot import anomalies_histogram_plot, auc_roc_plot, anomaly_pairplot, auc_time_plot

EXPERIMENT_FILEPATH = "./experiments results/"
INLIERS_ONLY_EXP_FILEPATH = EXPERIMENT_FILEPATH + "inliers only/"
AUC_TIME_EXP_FILEPATH = EXPERIMENT_FILEPATH + "auc_time/"
PARAMETERS_EXP_FILEPATH = EXPERIMENT_FILEPATH + "parameters/"


def main():
    http_parameters = {
        "exp_name": "Http",
        "data_path": "../datasets/sklearn/http.csv",
        "prediction_column": "attack",
        # "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 200, 250, 300, 350],
        # "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        # "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        # "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
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
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 200, 250, 300, 350],
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
        # "df_sampling_ratios": [.2, .4, .6, .8, 1],
        "df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        # "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        # "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        # "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        # "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    gen2_parameters = {
        "exp_name": "Gen2",
        "data_path": "../datasets/sklearn/gen2.csv",
        "prediction_column": "label",
        # "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        # "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        # "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    gen3_parameters = {
        "exp_name": "Gen3",
        "data_path": "../datasets/sklearn/gen3.csv",
        "prediction_column": "label",
        # "df_features_ratios": [.2, .4, .6, .8, 1],
        # "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 75, 95, 120, 150, 180, 220, 280, 300, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        # "tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        # "pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
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
    # datasets_parameters.extend([gen2_parameters, http_parameters, credit_card_parameters, cancer_parameters, landsat_parameters,
    # gen1_parameters, gen3_parameters, wines_parameters,
    # iris_parameters, sa_parameters, smtp_parameters])
    #datasets_parameters.extend([gen3_parameters, iris_parameters])

    datasets_parameters.extend(
        [credit_card_parameters, sa_parameters])

    #df_inlier_exp = mean_experiment_inliers(datasets_parameters, nb_run=10)
    #df_inlier_exp.to_csv(f"{INLIERS_ONLY_EXP_FILEPATH}mean results.csv", index=False)

    df_auc_time_exp = mean_experiment_auc_time(datasets_parameters, nb_run=10)
    df_auc_time_exp.to_csv(f"{AUC_TIME_EXP_FILEPATH}mean results.csv", index=False)



def mean_experiment_inliers(datasets_parameters, nb_run: int):

    dataset_names = []
    mean_auc_inliers = []
    mean_auc = []

    for dataset_parameters in datasets_parameters:

        auc_results = []
        auc_inliers_results = []

        for i in range(nb_run):
            auc, auc_inlier = inliers_experiment(dataset_parameters, i)

            auc_results.append(auc)
            auc_inliers_results.append(auc_inlier)

        dataset_names.append(dataset_parameters["exp_name"])
        mean_auc_inliers.append(np.mean(auc_inliers_results))
        mean_auc.append(np.mean(auc_results))

    data = {'dataset': dataset_names,
            'without outliers': mean_auc,
            'with outliers': mean_auc_inliers
            }

    return pd.DataFrame(data)

def inliers_experiment(parameters: dict, current_run: int):
    try:
        train_X, eval_X, train_y, eval_y = preprocess_data(parameters["data_path"], parameters["prediction_column"])

        # Separating inliers from outliers in training data
        outlier_indexes = train_y[train_y == 1].index.values
        train_X_inliers = train_X.drop(outlier_indexes)

        exp_name = parameters['exp_name']

        # Experiment without outliers
        exp_name_inlier = f"{exp_name}_train_inliers"
        exp_parameters, forest_parameters, exp_results, exp_figures = iforest_experiment_data(
            train_X_inliers, eval_X, eval_y, exp_name_inlier, parameters["prediction_column"]
        )

        auc_inlier_result = exp_results['AUC']
        save_exp_data(f"iforest_run_{current_run}_{exp_name_inlier}", exp_parameters, exp_results, INLIERS_ONLY_EXP_FILEPATH,
                      model_infos=forest_parameters, exp_figures=exp_figures)

        # Experiments with outliers
        exp_parameters, forest_parameters, exp_results, exp_figures = iforest_experiment_data(
            train_X, eval_X, eval_y, exp_name, parameters["prediction_column"]
        )

        auc_result = exp_results['AUC']
        save_exp_data(f"iforest_run_{current_run}_{exp_name}", exp_parameters, exp_results, INLIERS_ONLY_EXP_FILEPATH,
                          model_infos=forest_parameters, exp_figures=exp_figures)

        return auc_result, auc_inlier_result

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"Error while processing {parameters['data_path']}: {e}")

def mean_experiment_auc_time(dataset_parameters, nb_run: int):
    dataset_names = []
    algo_means = {}

    for parameters in dataset_parameters:
        dataset_names.append(parameters["exp_name"])
        summary = {}

        for i in range(nb_run):
            results = auc_time_experiment(parameters, i)

            #Add iteratively values from the run before computing the mean
            for algo, metrics in results.items():
                if algo not in summary:
                    summary[algo] = {
                        'AUC': [],
                        'training time': [],
                        'evaluation time': [],
                        'total time': []
                    }

                if "AUC" in metrics:
                    summary[algo]["AUC"].append(metrics["AUC"])
                if "training time" in metrics:
                    summary[algo]["training time"].append(metrics["training time"])
                if "evaluation time" in metrics:
                    summary[algo]["evaluation time"].append(metrics["evaluation time"])
                if "total time" in metrics:
                    summary[algo]["total time"].append(metrics["total time"])

        #Compute mean value for each algorithms for the current dataset
        for algo, metrics in summary.items():
            if algo not in algo_means:
                algo_means[algo] = {
                    "AUC": [],
                    "training_time": [],
                    "evaluation_time": [],
                    "total_time": []
                }
            for metric, values in metrics.items():
                if values:
                    mean_value = np.mean(values)
                else:
                    mean_value = None

                if metric == "AUC":
                    algo_means[algo]["AUC"].append(mean_value)
                elif metric == "training time":
                    algo_means[algo]["training_time"].append(mean_value)
                elif metric == "evaluation time":
                    algo_means[algo]["evaluation_time"].append(mean_value)
                elif metric == "total time":
                    algo_means[algo]["total_time"].append(mean_value)

    #convert algo_means into a Dataframe
    df_parts = {k: pd.DataFrame(v) for k, v in algo_means.items()}
    df = pd.concat(df_parts, axis=1)
    df.insert(0, 'dataset_names', dataset_names)

    return df


def auc_time_experiment(parameters: dict, current_run: int = 1):

    train_X, eval_X, train_y, eval_y = preprocess_data(parameters["data_path"], parameters["prediction_column"])

    auc_time_results = {}

    ocsvm_results = run_ocsvm(train_X, eval_X, eval_y, parameters, current_run)
    auc_time_results.update(ocsvm_results)

    skl_iforest_results = run_skl_iforest(train_X, eval_X, eval_y, parameters, current_run)
    auc_time_results.update(skl_iforest_results)

    iforest_results = run_iforest(train_X, eval_X, eval_y, parameters, current_run)
    auc_time_results.update(iforest_results)

    lof_results = run_lof(train_X, eval_X, train_y, eval_y, parameters, current_run)
    auc_time_results.update(lof_results)

    return auc_time_results

def run_skl_iforest(train_X, eval_X, eval_y, parameters:dict, current_run: int):

    exp_parameters, model_infos, exp_results = skl_iforest(train_X, eval_X, eval_y, parameters["exp_name"],
                                                           parameters["prediction_column"])

    results = {"skl_iforest": {
        "AUC": exp_results["AUC"],
        "training time": exp_results["training_time"],
        "evaluation time": exp_results["evaluation_time"],
        "total time": exp_results["total_time"]}
    }

    save_exp_data(f"skl_iforest_{parameters['exp_name']}_run_{current_run}", exp_parameters, exp_results, AUC_TIME_EXP_FILEPATH,
                  model_infos=model_infos)

    return results

def run_iforest(train_X, eval_X, eval_y, parameters, current_run: int):

    exp_parameters, forest_parameters, exp_results, _ = iforest_experiment_data(train_X, eval_X, eval_y,
                                                                                parameters["exp_name"],
                                                                                parameters["prediction_column"],
                                                                                nb_tree=100)

    results = {"iforest": {
        "AUC": exp_results["AUC"],
        "training time": exp_results["training time"],
        "evaluation time": exp_results["evaluation time"],
        "total time": exp_results["total time"]}
    }

    save_exp_data(f"iforest_{parameters['exp_name']}_run_{current_run}", exp_parameters, exp_results, AUC_TIME_EXP_FILEPATH,
                  model_infos=forest_parameters)

    return results

def run_ocsvm(train_X, eval_X, eval_y, parameters: dict, current_run: int):

    exp_parameters, model_infos, exp_results = ocsvm(train_X, eval_X, eval_y, parameters["exp_name"],
                                                     parameters["prediction_column"])
    results = {"ocsvm": {
        "AUC": exp_results["AUC"],
        "training time": exp_results["training_time"],
        "evaluation time": exp_results["evaluation_time"],
        "total time": exp_results["total_time"]}
    }

    save_exp_data(f"ocsvm_{parameters['exp_name']}_run_{current_run}", exp_parameters, exp_results, AUC_TIME_EXP_FILEPATH,
                  model_infos=model_infos)

    return results

def run_lof(train_X, eval_X, train_y, eval_y, parameters: dict, current_run: int):

    # recombining dataframes for LOF experiment
    X = pd.concat([train_X, eval_X])
    y = pd.concat([train_y, eval_y])

    exp_parameters, model_infos, exp_results = lof(X, y, parameters["exp_name"], parameters["prediction_column"])
    results = {"lof": {
        "AUC": exp_results["AUC"],
        "total time": exp_results["total_time"]}
    }

    save_exp_data(f"lof_{parameters['exp_name']}_run_{current_run}", exp_parameters, exp_results, AUC_TIME_EXP_FILEPATH,
                  model_infos=model_infos)

    return results

def skl_iforest(train_X: DataFrame, eval_X: DataFrame, eval_y: Series, exp_name: str, pred_column: str,
                nb_tree: int = 100, subsampling_size: int = 256, df_sampling_ratio: float = 1,
                df_features_ratio: float = 1):
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
    model_infos = {
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

    return exp_parameters, model_infos, exp_results


def lof(X: DataFrame, y: Series, exp_name: str, pred_column: str, df_sampling_ratio: float = 1,
        df_features_ratio: float = 1):
    print(f"Running LOF {exp_name} experiment")

    start_time = time.time()
    print("create LOF model")
    clf = LocalOutlierFactor(n_neighbors=20)
    print("Begin training and evaluation")
    y_pred = clf.fit_predict(X)
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
    model_infos = {
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

    return exp_parameters, model_infos, exp_results


def ocsvm(train_X: DataFrame, eval_X: DataFrame, eval_y: Series, exp_name: str, pred_column: str,
          df_sampling_ratio: float = 1, df_features_ratio: float = 1):
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

    # print(scores)
    # print(pred_y)
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
    model_infos = {
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

    return exp_parameters, model_infos, exp_results


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
                exp_parameters, forest_parameters, exp_results, exp_figures = iforest_experiment_file(exp_name=exp_name, file_path=parameters["data_path"],
                                                      pred_column=parameters["prediction_column"],
                                                      desired_tpr=parameters.get("desired_tpr", .85),
                                                      auc_roc=parameters["auc_roc_plot"],
                                                      anomaly_histogram=parameters["anomaly_histogram_plot"],
                                                      pairplot=parameters["pair_plot"], plot_save=True,
                                                      **{param_name: value})

                save_exp_data(exp_name, exp_parameters, exp_results, PARAMETERS_EXP_FILEPATH, forest_parameters, exp_figures)

                auc_results.append(exp_results["AUC"])
                time_results.append(exp_results["total time"])
            fig = auc_time_plot(auc_results, time_results, param_name, parameters[param_key])
            fig_exp_name = parameters["exp_name"]
            save_exp_figure(fig, PARAMETERS_EXP_FILEPATH, f"{fig_exp_name}_auc-time_{param_name}")


def iforest_experiment_file(exp_name: str, file_path: str, pred_column: str, nb_tree: int = 20,
                            subsampling_size: int = 256,
                            pathlength_height_limit: int = 0,
                            df_sampling_ratio: float = 1, df_features_ratio: float = 1, desired_tpr: float = .9,
                            auc_roc: bool = False, anomaly_histogram: bool = False, pairplot: bool = False,
                            plot_save: bool = False):
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

    # add scores and predictions
    # eval_X['Anomaly score'] = scores
    # eval_X['Anomaly prediction'] = prediction_y

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

    forest_parameters = iforest.get_parameters()

    exp_results = {
        "evaluation dataset size": eval_y.shape[0],
        "AUC": auc,
        "F1 score": f1_score,
        "TPR": tpr[1],
        "FPR": fpr[1],
        "anomaly threshold": threshold,
        "training time": iforest.training_time,
        "evaluation time": iforest.evaluation_time,
        "total time": iforest.training_time + iforest.evaluation_time,
    }

    exp_figures = {}
    if plot_save:
        exp_figures = {
            "AUC-ROC figures": auc_roc_fig,
            "Anomalies histogram": anomalies_hist_fig,
            "Scatter matrix": pairplot_fig
        }

    plt.close()
    return exp_parameters, forest_parameters, exp_results, exp_figures



def save_exp_data(exp_title: str, exp_parameters: dict, exp_results: dict, save_path: str, model_infos: dict = None,
                  exp_figures: dict = None):
    new_folder_name = exp_title + " " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    folder_path = save_path + new_folder_name
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
