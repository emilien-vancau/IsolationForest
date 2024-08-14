import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from pandas import DataFrame, Series
from sklearn import metrics
from sklearn.ensemble import IsolationForest as IF
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.models.plot.Plot import auc_time_plot, auc_roc_plot, anomalies_histogram_plot, anomaly_pairplot, auc_dist_factor_plot, \
    auc_contamination_hlimit_plot
from src.models.utility.Utility import preprocess_data_file, INLIERS_ONLY_EXP_FILEPATH, AUC_TIME_EXP_FILEPATH, PARAMETERS_EXP_FILEPATH, \
    preprocess_data
from src.models.dataset.CreateDatasets import create_anomalies_clusters_dataset
from src.models.isolationForest.IsolationForest import IsolationForest


def save_exp_data(exp_title: str, exp_parameters: dict, exp_results: dict, save_path: str, model_infos: dict = None,
                  exp_figures: dict = None):
    """
    Saves experiment data, parameters, results, and optional figures to a specified directory.

    Parameters:
        exp_title (str): The title of the experiment. This will be used to create a unique folder name.
        exp_parameters (dict): A dictionary containing the parameters used in the experiment.
        exp_results (dict): A dictionary containing the results of the experiment.
        save_path (str): The base directory where the experiment data will be saved.
        model_infos (dict, optional): A dictionary containing information about the model used in the experiment (default is None).
        exp_figures (dict, optional): A dictionary where keys are figure names (or identifiers) and values are figure objects to be saved (default is None).

    Functionality:
        1. Creates a new folder under the specified `save_path`. The folder name is generated using the `exp_title` and the current date and time to ensure uniqueness.
        2. Saves the experiment's parameters, model information, and results into a JSON file named `experiment_data.json` within the newly created folder.
        3. If `exp_figures` is provided, it iterates through the dictionary and saves each figure using the `save_exp_figure` function, with the figure name as the filename.

    Returns:
        None
    """

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
    """
    Saves a given figure to a specified directory with a given filename.

    Parameters:
        figure (Figure): The figure object to be saved. This is typically a Matplotlib figure.
        path (str): The directory path where the figure should be saved.
        figure_name (str): The name to be used for the saved figure file.

    Functionality:
        1. Constructs the full file path by combining the provided directory path and figure name.
        2. Saves the figure at the constructed path using the `savefig` method of the figure object.

    Returns:
        None
    """

    figurepath = os.path.join(path, figure_name)
    figure.savefig(figurepath)


def mean_experiment_inliers(datasets_parameters, nb_run: int):
    """
    Runs multiple experiments on datasets, computes the mean Area Under the Curve (AUC) both with and without outliers, and returns the results as a DataFrame.

    Parameters:
        datasets_parameters (list): A list of dictionaries, where each dictionary contains the parameters for a specific dataset experiment.
        nb_run (int): The number of runs to perform for each dataset experiment.

    Functionality:
        1. For each dataset in `datasets_parameters`:
            a. Runs the `inliers_experiment` function `nb_run` times, which returns the AUC values with and without inliers (outliers).
            b. Collects the AUC results across all runs.
            c. Computes the mean AUC for both with and without inliers.
        2. Compiles the dataset names, mean AUC values without outliers, and mean AUC values with outliers into a dictionary.
        3. Converts the dictionary into a Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing three columns:
            - 'dataset': Names of the datasets.
            - 'without outliers': Mean AUC values computed across runs without outliers.
            - 'with outliers': Mean AUC values computed across runs with outliers.
    """

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
    """
    Conducts an experiment on a dataset to compare the performance of an Isolation Forest model with and without outliers in the training data.

    Parameters:
        parameters (dict): A dictionary containing the parameters required for the experiment, including:
            - "data_path" (str): The file path to the dataset.
            - "prediction_column" (str): The column name in the dataset that contains the target labels.
            - "exp_name" (str): The base name for the experiment.
        current_run (int): The current run number of the experiment, used to distinguish multiple runs.

    Functionality:
        1. Loads and preprocesses the dataset specified in `parameters["data_path"]`, splitting it into training and evaluation sets.
        2. Separates inliers (non-outliers) from outliers in the training data based on the target labels.
        3. Performs an Isolation Forest experiment on the inliers only, saving the results and calculating the AUC.
        4. Performs the same experiment on the full dataset (including outliers), saving the results and calculating the AUC.
        5. Saves experiment data and figures using the `save_exp_data` function for both the inliers-only and the full dataset experiments.
        6. Returns the AUC values for the experiments with and without outliers.

    Returns:
        tuple: A tuple containing:
            - auc_result (float): The AUC value for the experiment using the full dataset (with outliers).
            - auc_inlier_result (float): The AUC value for the experiment using only the inliers (without outliers).

    Exception Handling:
        - Catches and logs a `FileNotFoundError` if the dataset file is not found.
        - Catches and logs any other exceptions that occur during the experiment, including the dataset path in the log message.
    """

    try:
        train_X, eval_X, train_y, eval_y = preprocess_data_file(parameters["data_path"], parameters["prediction_column"])

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
        save_exp_data(f"iforest_run_{current_run}_{exp_name_inlier}", exp_parameters, exp_results,
                      INLIERS_ONLY_EXP_FILEPATH,
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
    """
    Runs multiple experiments on datasets, computes the mean AUC and various time metrics (training, evaluation, total) for different algorithms, and returns the results as a DataFrame.

    Parameters:
        dataset_parameters (list): A list of dictionaries, where each dictionary contains the parameters for a specific dataset experiment.
        nb_run (int): The number of runs to perform for each dataset experiment.

    Functionality:
        1. Initializes lists and dictionaries to store dataset names and algorithm performance metrics.
        2. Iterates over each dataset in `dataset_parameters`:
            a. Adds the dataset name to the `dataset_names` list.
            b. Initializes a `summary` dictionary to store the AUC and time metrics for each algorithm across multiple runs.
            c. For each run:
                - Executes the `auc_time_experiment` function, which returns a dictionary of results (including AUC, training time, evaluation time, and total time) for each algorithm.
                - Updates the `summary` dictionary with the metrics from the current run.
            d. Computes the mean values of the metrics for each algorithm after all runs are completed and stores them in `algo_means`.
        3. Converts the `algo_means` dictionary into a Pandas DataFrame, where each algorithm's metrics are stored in separate columns.
        4. Inserts the dataset names as the first column in the DataFrame.
        5. Returns the resulting DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the mean AUC, training time, evaluation time, and total time for each algorithm, with the dataset names as the first column.

    Note:
        - The returned DataFrame has a hierarchical structure, with columns for each algorithm and sub-columns for each metric.
    """

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
    """
    Runs an experiment to evaluate different anomaly detection algorithms on a dataset, measuring their AUC (Area Under the Curve) and various time metrics (training, evaluation).

    Parameters:
        parameters (dict): A dictionary containing the parameters required for the experiment, including:
            - "data_path" (str): The file path to the dataset.
            - "prediction_column" (str): The column name in the dataset that contains the target labels.
        current_run (int, optional): The current run number of the experiment, used to distinguish multiple runs (default is 1).

    Functionality:
        1. Loads and preprocesses the dataset specified in `parameters["data_path"]`, splitting it into training and evaluation sets.
        2. Initializes an empty dictionary `auc_time_results` to store the results of the experiment.
        3. Executes different anomaly detection algorithms on the dataset, such as:
            - `run_skl_iforest`: Runs the Scikit-Learn Isolation Forest algorithm and records the AUC and time metrics.
            - `run_iforest`: Runs another Isolation Forest algorithm (potentially with custom or different settings) and records the AUC and time metrics.
        4. Updates the `auc_time_results` dictionary with the results from each algorithm.
        5. Returns the `auc_time_results` dictionary, which contains the AUC and time metrics for each algorithm.

    Returns:
        dict: A dictionary where keys are algorithm names and values are dictionaries containing the following metrics:
            - "AUC" (float): The AUC score for the algorithm.
            - "training time" (float): The time taken to train the model.
            - "evaluation time" (float): The time taken to evaluate the model.
            - "total time" (float): The sum of training and evaluation times.

    Note:
        - The code contains commented-out sections for running other algorithms like OCSVM (`run_ocsvm`) and LOF (`run_lof`), which can be activated as needed.
    """

    train_X, eval_X, train_y, eval_y = preprocess_data_file(parameters["data_path"], parameters["prediction_column"])

    auc_time_results = {}

    # ocsvm_results = run_ocsvm(train_X, eval_X, eval_y, parameters, current_run)
    # auc_time_results.update(ocsvm_results)

    skl_iforest_results = run_skl_iforest(train_X, eval_X, eval_y, parameters, current_run)
    auc_time_results.update(skl_iforest_results)

    iforest_results = run_iforest(train_X, eval_X, eval_y, parameters, current_run)
    auc_time_results.update(iforest_results)

    # lof_results = run_lof(train_X, eval_X, train_y, eval_y, parameters, current_run)
    # auc_time_results.update(lof_results)

    return auc_time_results

#Masking experiment
def mean_experiment_hlimit(exp_parameters: dict, height_limit: int, nb_run: int = 1):
    """
    Conducts an experiment to evaluate the impact of different height limits on the AUC (Area Under the Curve) scores across various contamination levels and returns the mean AUCs for each height limit.

    Parameters:
        exp_parameters (dict): A dictionary containing the parameters required for the experiment, including:
            - "exp_name" (str): The name of the experiment.
            - "nb_points" (int): The number of data points to generate in the synthetic dataset.
            - "contamination_ratios" (list): A list of contamination ratios to evaluate.
            - "nb_anomaly_clusters" (int): The number of anomaly clusters to generate in the dataset.
            - "distance_factor" (float): The factor that determines the distance of anomalies from normal data.
            - "anomaly_spread" (float): The spread (variance) of anomaly clusters.
        height_limit (int): The maximum height limit to evaluate in the experiment.
        nb_run (int, optional): The number of runs to perform for each combination of contamination ratio and height limit (default is 1).

    Functionality:
        1. Initializes lists to store mean AUC scores for each height limit and defines the range of height limits to evaluate.
        2. For each height limit:
            a. Initializes a list to store mean AUC scores for different contamination ratios at the current height limit.
            b. For each contamination ratio in `exp_parameters["contamination_ratios"]`:
                - Generates a synthetic dataset with specified anomaly clusters and contamination ratio using `create_anomalies_clusters_dataset`.
                - Updates `exp_parameters` with the default prediction column ("labels").
                - Runs the `hlimit_contamination_experiment` multiple times (determined by `nb_run`) for the current height limit and contamination ratio, storing the AUC scores.
                - Computes the mean AUC score across all runs for the current contamination ratio and appends it to the list for the current height limit.
            c. Appends the mean AUC scores for all contamination ratios at the current height limit to the `mean_hlimits_aucs` list.
        3. Plots the mean AUCs against contamination ratios for different height limits using `auc_contamination_hlimit_plot` and saves the figure using `save_exp_figure`.
        4. Returns the list `mean_hlimits_aucs`, containing the mean AUCs for each height limit across different contamination ratios.

    Returns:
        list: A list of lists, where each sublist contains the mean AUC scores for a specific height limit across all contamination ratios.

    """

    mean_hlimits_aucs = []
    height_limits = range(1, height_limit + 1)

    for hl in height_limits:

        mean_auc_cont_hl = []

        for ratio in exp_parameters["contamination_ratios"]:

            df = create_anomalies_clusters_dataset(nb_points=exp_parameters["nb_points"],
                                                   contamination_ratio=ratio,
                                                   nb_anomaly_clusters=exp_parameters["nb_anomaly_clusters"],
                                                   anomalies_distance_factor=exp_parameters["distance_factor"],
                                                   anomaly_spread=exp_parameters["anomaly_spread"])

            exp_parameters.update({"prediction_column": "labels"})  # default prediction column name for this experiment

            auc_results = []

            for i in range(nb_run):
                auc = hlimit_contamination_experiment(df, exp_parameters["exp_name"], exp_parameters["prediction_column"], i, ratio, hl)
                auc_results.append(auc)

            #mean for one contamination level in one hlimit
            mean_auc_cont_hl.append(np.mean(auc_results))

        #add aucs results for the current hlimit
        mean_hlimits_aucs.append(mean_auc_cont_hl)

    fig = auc_contamination_hlimit_plot(exp_parameters["contamination_ratios"], mean_hlimits_aucs, height_limits)
    save_exp_figure(fig, PARAMETERS_EXP_FILEPATH, f"{exp_parameters['exp_name']}_contamination_hlimit")

    return mean_hlimits_aucs

#Swamping experiment
def mean_experiment_dist_factor(exp_parameters: dict, nb_run: int = 1):
    """
    Conducts an experiment to evaluate the impact of different anomaly distance factors on the AUC (Area Under the Curve) scores and returns the mean AUCs for each distance factor.

    Parameters:
        exp_parameters (dict): A dictionary containing the parameters required for the experiment, including:
            - "exp_name" (str): The name of the experiment.
            - "nb_points" (int): The number of data points to generate in the synthetic dataset.
            - "contamination_ratio" (float): The ratio of anomalies to normal data in the dataset.
            - "nb_anomaly_clusters" (int): The number of anomaly clusters to generate in the dataset.
            - "distance_factors" (list): A list of distance factors to evaluate, which determine the distance of anomalies from normal data.
            - "anomaly_spread" (float): The spread (variance) of anomaly clusters.
        nb_run (int, optional): The number of runs to perform for each distance factor (default is 1).

    Functionality:
        1. Initializes a list `mean_auc` to store the mean AUC scores for each distance factor.
        2. Iterates over each distance factor in `exp_parameters["distance_factors"]`:
            a. Generates a synthetic dataset with the specified distance factor using `create_anomalies_clusters_dataset`.
            b. Updates `exp_parameters` with the default prediction column ("labels").
            c. Runs the `dist_factor_experiment` function multiple times (determined by `nb_run`) for the current distance factor, storing the AUC scores.
            d. Computes the mean AUC score across all runs for the current distance factor and appends it to the `mean_auc` list.
        3. Plots the mean AUCs against the distance factors using `auc_dist_factor_plot` and saves the figure using `save_exp_figure`.
        4. Returns the list `mean_auc`, containing the mean AUCs for each distance factor.

    Returns:
        list: A list containing the mean AUC scores for each distance factor in `exp_parameters["distance_factors"]`.

    """

    mean_auc = []

    for factor in exp_parameters["distance_factors"]:

        df = create_anomalies_clusters_dataset(nb_points=exp_parameters["nb_points"], contamination_ratio=exp_parameters["contamination_ratio"], nb_anomaly_clusters=exp_parameters["nb_anomaly_clusters"],
                                                           anomalies_distance_factor=factor, anomaly_spread=exp_parameters["anomaly_spread"])

        exp_parameters.update({"prediction_column": "labels"}) #default prediction column name for this experiment

        auc_results = []

        for i in range(nb_run):
            auc = dist_factor_experiment(df, exp_parameters["exp_name"], exp_parameters["prediction_column"], factor, i)
            auc_results.append(auc)

        mean_auc.append(np.mean(auc_results))

    fig = auc_dist_factor_plot(mean_auc, exp_parameters["distance_factors"])

    save_exp_figure(fig, PARAMETERS_EXP_FILEPATH, f"{exp_parameters['exp_name']}_distance_factors")

    return mean_auc


def hlimit_contamination_experiment(df: DataFrame, exp_name: str, prediction_column: str, current_run: int, current_contamination_ratio: int, current_hlimit: int):
    """
    Preprocesses data, runs an Isolation Forest experiment with specified parameters, and saves the results.

    Parameters:
        df (pd.DataFrame): The dataset to be processed.
        prediction_column (str): The column in the dataset that contains the target labels.
        exp_name (str): The name of the experiment.
        current_hlimit (int): The current height limit used in the Isolation Forest model.
        current_contamination_ratio (float): The contamination ratio of anomalies in the dataset.
        current_run (int): The current run number of the experiment.

    Functionality:
        1. Preprocesses the dataset by splitting it into training and evaluation sets using `preprocess_data`.
        2. Runs the Isolation Forest experiment using the `iforest_experiment_data` function with the provided parameters, including the height limit.
        3. Saves the experiment parameters, results, and model information using the `save_exp_data` function.
        4. Returns the AUC (Area Under the Curve) score from the experiment results.

    Returns:
        float: The AUC score obtained from the experiment.

    Note:
        - The experiment results are saved with a filename that includes the experiment name, contamination ratio, height limit, and run number for easy identification.
    """

    train_X, eval_X, _, eval_y = preprocess_data(df, prediction_column)

    exp_parameters, forest_parameters, exp_results, _ = iforest_experiment_data(train_X, eval_X, eval_y, exp_name, prediction_column,
                                                                                pathlength_height_limit=current_hlimit)

    save_exp_data(f"iforest_{exp_name}-contamination_{current_contamination_ratio}-hlimit_{current_hlimit}-run_{current_run}", exp_parameters, exp_results,
                  PARAMETERS_EXP_FILEPATH,
                  model_infos=forest_parameters)

    return exp_results["AUC"]

def dist_factor_experiment(df: DataFrame, exp_name: str, prediction_column: str, current_factor: int,  current_run: int):
    """
    Conducts an experiment to evaluate the Isolation Forest algorithm with a specified distance factor and saves the results.

    Parameters:
        df (pd.DataFrame): The dataset to be processed.
        exp_name (str): The name of the experiment.
        prediction_column (str): The column in the dataset that contains the target labels.
        current_factor (int): The current distance factor used in the Isolation Forest model.
        current_run (int): The current run number of the experiment.

    Functionality:
        1. Preprocesses the dataset by splitting it into training and evaluation sets using the `preprocess_data` function.
        2. Runs the Isolation Forest experiment using the `iforest_experiment_data` function with the provided distance factor and other parameters.
        3. Saves the experiment parameters, results, and model information using the `save_exp_data` function. The saved file includes details such as the experiment name, distance factor, and run number.
        4. Returns the AUC (Area Under the Curve) score obtained from the experiment results.

    Returns:
        float: The AUC score obtained from the Isolation Forest experiment.

    Note:
        - The file name for saving the experiment data includes the experiment name, distance factor, and run number for identification purposes.
    """

    train_X, eval_X, _, eval_y = preprocess_data(df, prediction_column)

    exp_parameters, forest_parameters, exp_results, _ = iforest_experiment_data(train_X, eval_X, eval_y, exp_name, prediction_column)

    save_exp_data(f"iforest_{exp_name}-dist_factor_{current_factor}-run_{current_run}", exp_parameters, exp_results,
                  PARAMETERS_EXP_FILEPATH,
                  model_infos=forest_parameters)

    return exp_results["AUC"]



def run_skl_iforest(train_X, eval_X, eval_y, parameters:dict, current_run: int):
    """
    Runs an Isolation Forest experiment using the Scikit-Learn implementation, records performance metrics, and saves the results.

    Parameters:
        train_X (pd.DataFrame): The training feature set.
        eval_X (pd.DataFrame): The evaluation feature set.
        eval_y (pd.Series): The evaluation target labels.
        parameters (dict): A dictionary containing parameters for the experiment, including:
            - "exp_name" (str): The name of the experiment.
            - "prediction_column" (str): The column in the dataset that contains the target labels.
        current_run (int): The current run number of the experiment.

    Functionality:
        1. Runs the Isolation Forest experiment using the Scikit-Learn implementation (`skl_iforest` function) with the provided parameters.
        2. Collects the experiment parameters, model information, and results, which include:
            - AUC (Area Under the Curve) score.
            - Training time.
            - Evaluation time.
            - Total time.
        3. Constructs a results dictionary with the performance metrics under the key `"skl_iforest"`.
        4. Saves the experiment parameters, results, and model information using the `save_exp_data` function. The saved file is named using the experiment name and run number for identification.
        5. Returns the results dictionary containing the metrics.

    Returns:
        dict: A dictionary with the following structure:
            - "skl_iforest": A dictionary containing:
                - "AUC" (float): The AUC score of the Isolation Forest model.
                - "training time" (float): The time taken to train the model.
                - "evaluation time" (float): The time taken to evaluate the model.
                - "total time" (float): The sum of training and evaluation times.
    """

    exp_parameters, model_infos, exp_results = skl_iforest(train_X, eval_X, eval_y, parameters["exp_name"],
                                                           parameters["prediction_column"])

    results = {"skl_iforest": {
        "AUC": exp_results["AUC"],
        "training time": exp_results["training_time"],
        "evaluation time": exp_results["evaluation_time"],
        "total time": exp_results["total_time"]}
    }

    save_exp_data(f"skl_iforest_{parameters['exp_name']}_run_{current_run}", exp_parameters, exp_results,
                  AUC_TIME_EXP_FILEPATH,
                  model_infos=model_infos)

    return results


def run_iforest(train_X, eval_X, eval_y, parameters, current_run: int, nb_tree: int = 100, subsampling_size: int = 256, pathlength_height_limit: int = 0):
    """
    Runs an Isolation Forest experiment with specified parameters, records performance metrics, and saves the results.

    Parameters:
        train_X (pd.DataFrame): The training feature set.
        eval_X (pd.DataFrame): The evaluation feature set.
        eval_y (pd.Series): The evaluation target labels.
        parameters (dict): A dictionary containing parameters for the experiment, including:
            - "exp_name" (str): The name of the experiment.
            - "prediction_column" (str): The column in the dataset that contains the target labels.
        current_run (int): The current run number of the experiment.
        nb_tree (int, optional): The number of trees in the Isolation Forest model (default is 100). This is the number of trees used in the ensemble for anomaly detection.
        subsampling_size (int, optional): The size of the subsample drawn from the training data to fit each tree (default is 256). This is the number of samples used to train each individual tree.
        pathlength_height_limit (int, optional): The height limit for path lengths in the Isolation Forest (default is 0). This parameter controls the maximum height of the trees.

    Functionality:
        1. Runs the Isolation Forest experiment using the `iforest_experiment_data` function with the provided parameters, including the number of trees, subsampling size, and pathlength height limit.
        2. Collects the experiment parameters, model information, and results, which include:
            - AUC (Area Under the Curve) score.
            - Training time.
            - Evaluation time.
            - Total time.
        3. Constructs a results dictionary with the performance metrics under the key `"iforest"`.
        4. Saves the experiment parameters, results, and model information using the `save_exp_data` function. The saved file is named using the experiment name and run number for identification.
        5. Returns the results dictionary containing the metrics.

    Returns:
        dict: A dictionary with the following structure:
            - "iforest": A dictionary containing:
                - "AUC" (float): The AUC score of the Isolation Forest model.
                - "training time" (float): The time taken to train the model.
                - "evaluation time" (float): The time taken to evaluate the model.
                - "total time" (float): The sum of training and evaluation times.
    """

    exp_parameters, forest_parameters, exp_results, _ = iforest_experiment_data(train_X, eval_X, eval_y,
                                                                                parameters["exp_name"],
                                                                                parameters["prediction_column"],
                                                                                nb_tree=nb_tree, #default tree numbers in original paper experiment
                                                                                subsampling_size= subsampling_size,
                                                                                pathlength_height_limit=pathlength_height_limit)

    results = {"iforest": {
        "AUC": exp_results["AUC"],
        "training time": exp_results["training time"],
        "evaluation time": exp_results["evaluation time"],
        "total time": exp_results["total time"]}
    }

    save_exp_data(f"iforest_{parameters['exp_name']}_run_{current_run}", exp_parameters, exp_results,
                  AUC_TIME_EXP_FILEPATH,
                  model_infos=forest_parameters)

    return results


def run_ocsvm(train_X, eval_X, eval_y, parameters: dict, current_run: int):
    """
    Runs an One-Class Support Vector Machine (OCSVM) experiment, records performance metrics, and saves the results.

    Parameters:
        train_X (pd.DataFrame): The training feature set.
        eval_X (pd.DataFrame): The evaluation feature set.
        eval_y (pd.Series): The evaluation target labels.
        parameters (dict): A dictionary containing parameters for the experiment, including:
            - "exp_name" (str): The name of the experiment.
            - "prediction_column" (str): The column in the dataset that contains the target labels.
        current_run (int): The current run number of the experiment.

    Functionality:
        1. Runs the OCSVM experiment using the `ocsvm` function with the provided parameters.
        2. Collects the experiment parameters, model information, and results, which include:
            - AUC (Area Under the Curve) score.
            - Training time.
            - Evaluation time.
            - Total time.
        3. Constructs a results dictionary with the performance metrics under the key `"ocsvm"`.
        4. Saves the experiment parameters, results, and model information using the `save_exp_data` function. The saved file is named using the experiment name and run number for identification.
        5. Returns the results dictionary containing the metrics.

    Returns:
        dict: A dictionary with the following structure:
            - "ocsvm": A dictionary containing:
                - "AUC" (float): The AUC score of the OCSVM model.
                - "training time" (float): The time taken to train the model.
                - "evaluation time" (float): The time taken to evaluate the model.
                - "total time" (float): The sum of training and evaluation times.
    """

    exp_parameters, model_infos, exp_results = ocsvm(train_X, eval_X, eval_y, parameters["exp_name"],
                                                     parameters["prediction_column"])
    results = {"ocsvm": {
        "AUC": exp_results["AUC"],
        "training time": exp_results["training_time"],
        "evaluation time": exp_results["evaluation_time"],
        "total time": exp_results["total_time"]}
    }

    save_exp_data(f"ocsvm_{parameters['exp_name']}_run_{current_run}", exp_parameters, exp_results,
                  AUC_TIME_EXP_FILEPATH,
                  model_infos=model_infos)

    return results


def run_lof(train_X, eval_X, train_y, eval_y, parameters: dict, current_run: int):
    """
    Runs a Local Outlier Factor (LOF) experiment, combines the training and evaluation datasets, records performance metrics, and saves the results.

    Parameters:
        train_X (pd.DataFrame): The training feature set.
        eval_X (pd.DataFrame): The evaluation feature set.
        train_y (pd.Series): The training target labels.
        eval_y (pd.Series): The evaluation target labels.
        parameters (dict): A dictionary containing parameters for the experiment, including:
            - "exp_name" (str): The name of the experiment.
            - "prediction_column" (str): The column in the dataset that contains the target labels.
        current_run (int): The current run number of the experiment.

    Functionality:
        1. Combines the training and evaluation datasets (`train_X` and `eval_X`) into a single feature set (`X`), and the corresponding target labels (`train_y` and `eval_y`) into a single label set (`y`).
        2. Runs the LOF experiment using the combined dataset and the `lof` function with the provided parameters.
        3. Collects the experiment parameters, model information, and results, which include:
            - AUC (Area Under the Curve) score.
            - Total time taken for the experiment.
        4. Constructs a results dictionary with the performance metrics under the key `"lof"`.
        5. Saves the experiment parameters, results, and model information using the `save_exp_data` function. The saved file is named using the experiment name and run number for easy identification.
        6. Returns the results dictionary containing the metrics.

    Returns:
        dict: A dictionary with the following structure:
            - "lof": A dictionary containing:
                - "AUC" (float): The AUC score of the LOF model.
                - "total time" (float): The total time taken for the LOF experiment.
    """

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
    """
    Trains and evaluates an Isolation Forest model using the Scikit-Learn implementation, and returns the experiment parameters, model information, and results.

    Parameters:
        train_X (pd.DataFrame): The feature set for training the model.
        eval_X (pd.DataFrame): The feature set for evaluating the model.
        eval_y (pd.Series): The target labels for evaluating the model.
        exp_name (str): The name of the experiment.
        pred_column (str): The column name in the dataset that contains the target labels.
        nb_tree (int, optional): The number of trees in the Isolation Forest model (default is 100). This is the number of trees used in the ensemble.
        subsampling_size (int, optional): The number of samples used to fit each tree (default is 256). This controls the size of the sample drawn from the training data to fit each individual tree.
        df_sampling_ratio (float, optional): The ratio of samples to use from the dataset during preprocessing (default is 1). This parameter controls the proportion of the dataset used.
        df_features_ratio (float, optional): The ratio of features to use from the dataset during preprocessing (default is 1). This parameter controls the proportion of features used.

    Returns:
        tuple: A tuple containing three dictionaries:
            - exp_parameters (dict): A dictionary containing the parameters of the experiment, including:
                - "experiment_name": The name of the experiment.
                - "prediction feature": The name of the column used for predictions.
                - "pre_processing_sampling_ratio": The ratio of samples used for preprocessing.
                - "pre_processing_feature_ratio": The ratio of features used for preprocessing.
                - "training_size": The number of samples in the training dataset.
                - "evaluation_size": The number of samples in the evaluation dataset.
            - model_infos (dict): A dictionary containing information about the trained model, including:
                - "model name": The name of the model ("sklearn iforest").
                - "n_estimators": The number of trees in the forest.
                - "contamination": The proportion of outliers in the dataset (if set).
                - "max_samples": The number of samples used to fit each tree.
                - "max_features": The number of features to draw from the dataset (if set).
                - "bootstrap": Whether bootstrap samples are used (if set).
                - "random_state": The seed used by the random number generator.
                - "verbose": The verbosity level of the model.
                - "n_jobs": The number of jobs to run in parallel.
                - "warm_start": Whether to reuse the solution of the previous call to fit.
            - exp_results (dict): A dictionary containing the results of the experiment, including:
                - "evaluation dataset size": The number of samples in the evaluation dataset.
                - "AUC": The Area Under the Curve score.
                - "F1_score": The F1 score of the model.
                - "training_time": The time taken to train the model (in seconds).
                - "evaluation_time": The time taken to evaluate the model (in seconds).
                - "total_time": The total time taken for training and evaluation (in seconds).

    Note:
        - The function prints the training time, evaluation time, total time, AUC score, and F1 score for the Isolation Forest model.
    """

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
    """
    Trains and evaluates a Local Outlier Factor (LOF) model, and returns the experiment parameters, model information, and results.

    Parameters:
        X (pd.DataFrame): The feature set for training and evaluation of the LOF model.
        y (pd.Series): The target labels for training and evaluation of the LOF model.
        exp_name (str): The name of the experiment.
        pred_column (str): The column name in the dataset that contains the target labels.
        df_sampling_ratio (float, optional): The ratio of samples to use from the dataset during preprocessing (default is 1). This parameter controls the proportion of the dataset used.
        df_features_ratio (float, optional): The ratio of features to use from the dataset during preprocessing (default is 1). This parameter controls the proportion of features used.

    Returns:
        tuple: A tuple containing three dictionaries:
            - exp_parameters (dict): A dictionary containing the parameters of the experiment, including:
                - "experiment_name": The name of the experiment.
                - "prediction_feature": The name of the column used for predictions.
                - "pre_processing_sampling_size": The ratio of samples used for preprocessing.
                - "pre_processing_feature_size": The ratio of features used for preprocessing.
            - model_infos (dict): A dictionary containing information about the trained model, including:
                - "model_name": The name of the model ("LOF").
                - "n_neighbors": The number of neighbors used by the LOF model.
                - "contamination": The proportion of outliers in the dataset (if set).
                - "algorithm": The algorithm used to compute the LOF (default is 'auto').
                - "leaf_size": The leaf size parameter for the LOF algorithm.
                - "metric": The distance metric used by the LOF model.
                - "metric_params": Parameters for the distance metric (if any).
                - "p": The power parameter for the Minkowski distance metric (if used).
                - "n_jobs": The number of parallel jobs to run for computation (if applicable).
                - "novelty": Whether the model is used for novelty detection (if set).
            - exp_results (dict): A dictionary containing the results of the experiment, including:
                - "dataset size": The number of samples in the dataset.
                - "AUC": The Area Under the Curve score.
                - "F1_score": The F1 score of the model.
                - "total_time": The total time taken for training and evaluation (in seconds).

    Note:
        - The function prints the total time taken, AUC score, and F1 score for the LOF model.
    """

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
    """
    Trains and evaluates a One-Class Support Vector Machine (SVM) model, and returns the experiment parameters, model information, and results.

    Parameters:
        train_X (pd.DataFrame): The feature set used for training the One-Class SVM model.
        eval_X (pd.DataFrame): The feature set used for evaluating the One-Class SVM model.
        eval_y (pd.Series): The target labels for evaluating the One-Class SVM model.
        exp_name (str): The name of the experiment.
        pred_column (str): The column name in the dataset that contains the target labels.
        df_sampling_ratio (float, optional): The ratio of samples to use from the dataset during preprocessing (default is 1). This parameter controls the proportion of the dataset used.
        df_features_ratio (float, optional): The ratio of features to use from the dataset during preprocessing (default is 1). This parameter controls the proportion of features used.

    Returns:
        tuple: A tuple containing three dictionaries:
            - exp_parameters (dict): A dictionary containing the parameters of the experiment, including:
                - "experiment_name": The name of the experiment.
                - "prediction_feature": The name of the column used for predictions.
                - "pre_processing_sampling_ratio": The ratio of samples used for preprocessing.
                - "pre_processing_feature_ratio": The ratio of features used for preprocessing.
                - "training_size": The number of samples in the training dataset.
                - "evaluation_size": The number of samples in the evaluation dataset.
            - model_infos (dict): A dictionary containing information about the trained model, including:
                - "model name": The name of the model ("one class SVM").
                - "kernel": The kernel used by the One-Class SVM model.
                - "degree": The degree of the polynomial kernel function (if used).
                - "nu": The upper bound on the fraction of margin errors and lower bound on the fraction of support vectors (if set).
                - "gamma": The kernel coefficient for the 'rbf', 'poly', and 'sigmoid' kernels.
                - "coef0": The coefficient for the 'poly' and 'sigmoid' kernels.
                - "tol": The tolerance for stopping criteria.
                - "shrinking": Whether to use the shrinking heuristic.
                - "verbose": Whether to enable verbose output.
                - "cache_size": The size of the kernel cache (in MB).
                - "max_iter": The maximum number of iterations for the solver.
            - exp_results (dict): A dictionary containing the results of the experiment, including:
                - "evaluation_dataset_size": The number of samples in the evaluation dataset.
                - "AUC": The Area Under the Curve score.
                - "F1_score": The F1 score of the model.
                - "training_time": The time taken to train the model (in seconds).
                - "evaluation_time": The time taken to evaluate the model (in seconds).
                - "total_time": The total time taken for training and evaluation (in seconds).

    Note:
        - The function prints the training time, evaluation time, total time, AUC score, and F1 score for the One-Class SVM model.
    """

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


def run_iforest_param_experiments(parameters: dict, nb_run: int = 1):
    """
    Runs a series of experiments to evaluate the impact of different hyperparameters on the performance of the Isolation Forest algorithm.

    This function iterates over a set of hyperparameters, conducts experiments for each parameter value, and saves the results. It also generates and saves plots showing the relationship between the hyperparameters and performance metrics.

    Parameters:
        parameters (dict): A dictionary containing the experiment parameters and configurations. It should include:
            - "exp_name" (str): The base name for the experiment.
            - "data_path" (str): The path to the dataset file.
            - "prediction_column" (str): The name of the column in the dataset that contains the target labels.
            - "auc_roc_plot" (bool): Whether to generate and save the ROC curve plot.
            - "anomaly_histogram_plot" (bool): Whether to generate and save the anomaly histogram plot.
            - "pair_plot" (bool): Whether to generate and save the pair plot.
            - "df_sampling_ratios" (list of float, optional): A list of sampling ratios for the dataset.
            - "df_features_ratios" (list of float, optional): A list of feature ratios for the dataset.
            - "forest_nb_trees" (list of int, optional): A list of numbers of trees to use in the Isolation Forest model.
            - "forest_subsampling_sizes" (list of int, optional): A list of subsampling sizes to use in the Isolation Forest model.
            - "pathlength_height_limit" (list of int, optional): A list of height limits for the path length of the Isolation Forest model.

        nb_run (int, optional): The number of times to repeat each experiment with different parameter values. Default is 1.

    Returns:
        None: The function saves experiment data and figures to the specified file paths. It does not return any values.

    Note:
        - For each hyperparameter, the function runs experiments with different values specified in the `parameters` dictionary.
        - Results for each parameter value are collected, and the mean AUC and total time are calculated across multiple runs.
        - A plot showing the relationship between AUC and total time is generated and saved for each hyperparameter.
        - The `iforest_experiment_file` function is called to perform the experiment with the current hyperparameter setting. The results are saved using the `save_exp_data` function.
        - The function generates and saves plots of AUC vs. total time for different hyperparameters. The plot is saved using the `save_exp_figure` function.
    """

    param_mappings = {
        "df_sampling_ratios": "df_sampling_ratio",
        "df_features_ratios": "df_features_ratio",
        "forest_nb_trees": "nb_tree",
        "forest_subsampling_sizes": "subsampling_size",
        "pathlength_height_limit": "pathlength_height_limit",
    }

    for param_key, param_name in param_mappings.items():
        if param_key in parameters:

            mean_auc_results = []
            mean_time_results = []
            fig = None

            for i in range(nb_run):

                print(f"Starting hyperparameter {param_name} - run {i} experiment")

                auc_results = []
                time_results = []
                for value in parameters[param_key]:
                    exp_name = parameters["exp_name"] + f" {param_name}-{value}-run_{i}"
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

                mean_auc_results.append(auc_results)
                mean_time_results.append(time_results)

            #create matrix and compute means
            mean_auc_matrix = np.vstack(mean_auc_results)
            mean_time_matrix = np.vstack(mean_time_results)
            mean_aucs = np.mean(mean_auc_matrix, axis=0)
            mean_times = np.mean(mean_time_matrix, axis=0)

            #AUC TIME PLOT

            if param_name == "subsampling_size":
                fig = auc_time_plot(mean_aucs, mean_times, param_name, parameters[param_key], x_log_display=True)
            else:
                fig = auc_time_plot(mean_aucs, mean_times, param_name, parameters[param_key])

            fig_exp_name = parameters["exp_name"]
            save_exp_figure(fig, PARAMETERS_EXP_FILEPATH, f"{fig_exp_name}_auc-time_{param_name}")


def iforest_experiment_file(exp_name: str, file_path: str, pred_column: str, nb_tree: int = 20,
                            subsampling_size: int = 256,
                            pathlength_height_limit: int = 0,
                            df_sampling_ratio: float = 1, df_features_ratio: float = 1, desired_tpr: float = .9,
                            auc_roc: bool = False, anomaly_histogram: bool = False, pairplot: bool = False,
                            plot_save: bool = False):
    """
    Runs an Isolation Forest experiment based on the provided parameters and optionally generates and saves plots.

    This function handles data preprocessing and invokes `iforest_experiment_data` to perform the core experiment with the Isolation Forest algorithm. It also manages the optional generation and saving of various plots.

    Parameters:
        exp_name (str): The name of the experiment, used to label files and plots.
        file_path (str): The path to the dataset file. The file should be in a format that can be processed by `preprocess_data_file`.
        pred_column (str): The name of the column in the dataset that contains the target labels.
        nb_tree (int, optional): Number of trees to use in the Isolation Forest model. Default is 20.
        subsampling_size (int, optional): Size of the random subsamples to draw from the training data. Default is 256.
        pathlength_height_limit (int, optional): Height limit for path length in the Isolation Forest model. Default is 0 (no limit).
        df_sampling_ratio (float, optional): The ratio of the dataset to use for training and evaluation. Default is 1 (use the entire dataset).
        df_features_ratio (float, optional): The ratio of features to use for training and evaluation. Default is 1 (use all features).
        desired_tpr (float, optional): Desired True Positive Rate for the experiment. Default is 0.9.
        auc_roc (bool, optional): Whether to generate and save the ROC curve plot. Default is False.
        anomaly_histogram (bool, optional): Whether to generate and save the histogram of anomalies. Default is False.
        pairplot (bool, optional): Whether to generate and save the pair plot of features. Default is False.
        plot_save (bool, optional): Whether to save the generated plots. Default is False.

    Returns:
        dict: A dictionary containing the experiment parameters, model information, and results.
              The dictionary is returned by `iforest_experiment_data` function.

    Note:
        - Data is preprocessed using `preprocess_data_file`, which handles data loading and preparation.
        - The `iforest_experiment_data` function is then called to perform the Isolation Forest experiment.
        - Plotting and saving options are controlled by the `auc_roc`, `anomaly_histogram`, `pairplot`, and `plot_save` parameters.
    """

    train_X, eval_X, train_y, eval_y = preprocess_data_file(file_path, pred_column, df_sampling_ratio, df_features_ratio)

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
    """
        Runs an Isolation Forest experiment, evaluates the results, and optionally generates and saves various plots.

        Parameters:
            - train_X (DataFrame): Training feature data.
            - eval_X (DataFrame): Evaluation feature data.
            - eval_y (Series): Evaluation target labels.
            - exp_name (str): Name of the experiment.
            - pred_column (str): Name of the prediction column.
            - nb_tree (int): Number of trees in the Isolation Forest.
            - subsampling_size (int): Size of the subsamples used.
            - pathlength_height_limit (int): Height limit for path length.
            - df_sampling_ratio (float): Ratio of rows to use from the dataset.
            - df_features_ratio (float): Ratio of features to use from the dataset.
            - desired_tpr (float): Desired True Positive Rate.
            - auc_roc (bool): Whether to generate the ROC curve plot.
            - anomaly_histogram (bool): Whether to generate the anomaly histogram.
            - pairplot (bool): Whether to generate the pair plot.
            - plot_save (bool): Whether to save the generated plots.

        Returns:
            - exp_parameters (dict): Dictionary of experiment parameters.
            - forest_parameters (dict): Dictionary of forest model parameters.
            - exp_results (dict): Dictionary of experiment results, including AUC, F1 score, and timing.
            - exp_figures (dict): Dictionary of generated figures (if any).
        """

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