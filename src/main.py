from src.Experiment import run_iforest_param_experiments


def main():
    http_parameters = {
        "exp_name": "Http",
        "data_path": "../datasets/sklearn/http.csv",
        "prediction_column": "attack",
        # "df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 200, 250, 300, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
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
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 75, 95, 120, 150, 180, 220, 280, 300, 350],
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
        #"df_sampling_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        #"df_features_ratios": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        "forest_nb_trees": [3, 5, 7, 9, 13, 17, 21, 25, 35, 55, 95, 150, 250, 350],
        "forest_subsampling_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        #"tree_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        #"pathlength_height_limit": [2, 4, 8, 16, 32, 64, 128, 256],
        "desired_tpr": .9,
        "auc_roc_plot": True,
        "anomaly_histogram_plot": True,
        "pair_plot": False
    }

    datasets_parameters = []
    datasets_parameters.extend([gen2_parameters, http_parameters, credit_card_parameters, cancer_parameters, landsat_parameters,
    gen1_parameters, gen3_parameters, wines_parameters,
    iris_parameters, sa_parameters, smtp_parameters])

    #datasets_parameters.extend(
    #    [gen1_parameters, gen2_parameters, cancer_parameters, landsat_parameters, wines_parameters])

    for dataset_parameters in datasets_parameters:
        run_iforest_param_experiments(dataset_parameters, 10)

    #df_inlier_exp = mean_experiment_inliers(datasets_parameters, nb_run=10)
    #df_inlier_exp.to_csv(f"{INLIERS_ONLY_EXP_FILEPATH}mean results.csv", index=False)

    #df_auc_time_exp = mean_experiment_auc_time(datasets_parameters, nb_run=10)
    #df_auc_time_exp.to_csv(f"{AUC_TIME_EXP_FILEPATH}mean results.csv", index=False)


if __name__ == "__main__":
    main()
