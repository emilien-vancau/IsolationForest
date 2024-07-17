import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

EXPERIMENT_FILEPATH = "./experiments results/"
INLIERS_ONLY_EXP_FILEPATH = EXPERIMENT_FILEPATH + "inliers only/"
AUC_TIME_EXP_FILEPATH = EXPERIMENT_FILEPATH + "auc_time/"
PARAMETERS_EXP_FILEPATH = EXPERIMENT_FILEPATH + "parameters/"

def get_size_ratio(size: int, ratio: float):
    if 0 < ratio <= 1:
        return round(size * ratio)
    print("ratio should be between 0 (exclude) and 1")
    return


def preprocess_data(file_path: str, prediction_column: str, sample_row_ratio=None, sample_column_ratio=None):
    df = pd.read_csv(file_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    #drop rows with missing values
    df = df.dropna()

    # drop column which contains categorical values
    proc_df = df[df.select_dtypes(include=['float64', 'int64']).columns.tolist()]

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
        #train_X, eval_X, train_y, eval_y = train_test_split(X, y, random_state=1)

        # Using StratifiedShuffleSplit to ensure both classes are present in train and eval sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for train_index, test_index in sss.split(X, y):
            train_X, eval_X = X.iloc[train_index], X.iloc[test_index]
            train_y, eval_y = y.iloc[train_index], y.iloc[test_index]

        return train_X, eval_X, train_y, eval_y

    else:
        # split input data
        train_X, eval_X = train_test_split(proc_df, random_state=1)
        return train_X, eval_X, train_y, eval_y


def drop_outliers(df: DataFrame, column: str):
    return df[df[column] != 1]

def display_class_proportions(df: DataFrame, label_column: str):
    df_proportions = df[label_column].value_counts(normalize=True)
    print(f"Classes proportions in {label_column} column")
    print(df_proportions)


def display_firsts_anomalies(df: DataFrame, n: int = 10):
    pass


