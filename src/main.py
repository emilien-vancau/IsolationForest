from src.models.experiment.Experiment import run_iforest_param_experiments, mean_experiment_auc_time, mean_experiment_inliers, \
    mean_experiment_dist_factor, mean_experiment_hlimit
from src.models.utility.Utility import AUC_TIME_EXP_FILEPATH, INLIERS_ONLY_EXP_FILEPATH, generate_geometric_sequence


def main():
    http_parameters = {
        "exp_name": "Http",
        "data_path": "../datasets/sklearn/http.csv",
        "prediction_column": "attack",
        # "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 200, 250, 300, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
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
        #"df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
        #"tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        #"pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    credit_card_parameters = {
        "exp_name": "cred_card_fraud",
        "data_path": "../datasets/kaggle/credit_card_fraud/creditcard.csv",
        "prediction_column": "Class",
        #"df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
        #"tree_height_limit": [4, 8, 16, 32, 64, 128],
        #"pathlength_height_limit": [4, 8, 16, 32, 64, 128],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    iris_parameters = {
        "exp_name": "Iris",
        "data_path": "../datasets/sklearn/iris.csv",
        "prediction_column": "class",
        #"df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
        #"tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        #"pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    wines_parameters = {
        "exp_name": "Wines",
        "data_path": "../datasets/sklearn/wines.csv",
        "prediction_column": "class",
        #"df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
        #"tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        #"pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
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
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
        #"tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        #"pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    sa_parameters = {
        "exp_name": "Sa",
        "data_path": "../datasets/sklearn/sa.csv",
        "prediction_column": "label",
        #"df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
        #"tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        #"pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
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
        #"df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
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
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
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
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
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
        #"df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": generate_geometric_sequence(15, 2),
        #"tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        #"pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    swamping_exp_parameters = {
        "exp_name": "swamping_effect",
        "distance_factors": [x / 10.0 for x in range(5, 35, 2)], #begin from 0.5 to 3.5 with 0.5 each steps
        "nb_points": 1500,
        "contamination_ratio": .1,
        "nb_anomaly_clusters": 3,
        "anomaly_spread": .2
    }

    masking_exp_parameters = {
        "exp_name": "masking_effect",
        "nb_points": 4096,
        "contamination_ratios": [x / 100.0 for x in range(10, 100, 5)],
        "distance_factor": 2,
        "nb_anomaly_clusters": 10,
        "anomaly_spread": .2
    }

    datasets_parameters = []
    datasets_parameters.extend([gen2_parameters, http_parameters, credit_card_parameters, cancer_parameters, landsat_parameters,
    gen1_parameters, gen3_parameters, wines_parameters,
    iris_parameters, sa_parameters, smtp_parameters])

    # Experiment hyperparameters
    for dataset_parameters in datasets_parameters:
        run_iforest_param_experiments(dataset_parameters, 10)

    # Training only with inliers experiment
    df_inlier_exp = mean_experiment_inliers(datasets_parameters, nb_run=10)
    df_inlier_exp.to_csv(f"{INLIERS_ONLY_EXP_FILEPATH}mean results.csv", index=False)

    df_auc_time_exp = mean_experiment_auc_time(datasets_parameters, nb_run=10)
    df_auc_time_exp.to_csv(f"{AUC_TIME_EXP_FILEPATH}mean results.csv", index=False)

    # Masking and Swamping measurements of IForest model
    mean_experiment_dist_factor(swamping_exp_parameters, 10)
    mean_experiment_hlimit(masking_exp_parameters, 6, 10)

if __name__ == "__main__":
    main()
