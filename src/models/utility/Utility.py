import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

#Constants filepaths
EXPERIMENT_FILEPATH = "../../experiments results/"
INLIERS_ONLY_EXP_FILEPATH = EXPERIMENT_FILEPATH + "inliers only/"
AUC_TIME_EXP_FILEPATH = EXPERIMENT_FILEPATH + "auc_time/"
PARAMETERS_EXP_FILEPATH = EXPERIMENT_FILEPATH + "parameters/"

def get_size_ratio(size: int, ratio: float):
    """
        Calculate a new size based on the original size and a ratio.

        Parameters:
            size (int): The original size.
            ratio (float): The ratio of the original size to use (between 0 and 1).

        Returns:
            int: The rounded new size if the ratio is valid.
            None: If the ratio is not between 0 and 1.
    """

    if 0 < ratio <= 1:
        return round(size * ratio)
    print("ratio should be between 0 (exclude) and 1")
    return

def generate_geometric_sequence(sequence_length: int = 15, start_index: int = 0):
    """
        Generate a geometric sequence of powers of 2.

        Parameters:
            sequence_length (int): Number of terms in the sequence.
            start_index (int): Starting index for the powers of 2.

        Returns:
            List[int]: A list of integers where each element is 2 raised to a power starting from start_index.
            None: If sequence_length is not positive.
    """

    if sequence_length > 0:
        return [2 ** i for i in range(start_index, start_index + sequence_length)]
    print("sequence number should be positive")
    return

def display_outliers_proportions(df: DataFrame, label_column: str):
    """
        Display the proportions of different classes in the specified column of a DataFrame.

        Parameters:
            df (DataFrame): The DataFrame containing the data.
            label_column (str): The column name for which to display the proportions of unique values.

        Returns:
            None: This function prints the proportions of unique values in the specified column.
    """

    print(df[label_column].value_counts(normalize=True))


def preprocess_data_file(file_path: str, prediction_column: str, sample_row_ratio=None, sample_column_ratio=None):
    """
        Load and preprocess data from a CSV file.

        Parameters:
            file_path (str): The path to the CSV file.
            prediction_column (str): The name of the column with the target variable or labels.
            sample_row_ratio (float, optional): Ratio of rows to sample. If None, all rows are used.
            sample_column_ratio (float, optional): Ratio of columns to sample. If None, all columns are used.

        Returns:
            DataFrames: Preprocessed training and evaluation sets, and Series: Labels.
    """

    df = pd.read_csv(file_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    return preprocess_data(df, prediction_column, sample_column_ratio, sample_row_ratio)


def preprocess_data(input_df: DataFrame, prediction_column: str, sample_column_ratio=None, sample_row_ratio=None):
    """
    This function preprocesses a given DataFrame by handling missing values, filtering out categorical columns,
    sampling rows and columns if specified, and optionally encoding a prediction column.
    It then splits the data into training and evaluation sets.

    :param input_df: The input DataFrame to be processed.
    :param prediction_column: The name of the column in input_df that is to be used as the target variable for prediction.
                                If specified, this column is included in the processed DataFrame and converted to a boolean format if it contains two unique values.
    :param sample_column_ratio: Ratio of columns to sample from the processed DataFrame. Should be between 0 and 1. If not provided, all columns are used.
    :param sample_row_ratio: Ratio of rows to sample from the processed DataFrame. Should be between 0 and 1. If not provided, all rows are used.

    :return:
        If prediction_column is specified:

            train_X (DataFrame): Features of the training set.
            eval_X (DataFrame): Features of the evaluation set.
            train_y (Series): Target values of the training set.
            eval_y (Series): Target values of the evaluation set.

        If prediction_column is not specified:

            train_X (DataFrame): Features of the training set.
            eval_X (DataFrame): Features of the evaluation set.
            train_y (None): Placeholder, always None.
            eval_y (None): Placeholder, always None.
    """

    # drop rows with missing values
    input_df = input_df.dropna()
    # drop column which contains categorical values
    proc_df = input_df[input_df.select_dtypes(include=['float64', 'int64']).columns.tolist()]
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
        proc_df[prediction_column] = input_df[prediction_column]

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
        # train_X, eval_X, train_y, eval_y = train_test_split(X, y, random_state=1)

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
    """
        Removes rows from a DataFrame where the value in a specified column is not equal to 1.

        :param df: input dataframe
        :param column: outlier label column
        :return: A dataframe without outliers

    """

    return df[df[column] != 1]

def display_class_proportions(df: DataFrame, label_column: str):
    """
    Prints the proportion of each class in the specified column of the DataFrame.

    This function calculates the normalized value counts of the unique values in the given column
    of the DataFrame and prints the proportions of each class.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    label_column (str): The name of the column for which class proportions are to be calculated.

    Returns:
    None: This function prints the class proportions and does not return any value.
    """

    df_proportions = df[label_column].value_counts(normalize=True)
    print(f"Classes proportions in {label_column} column")
    print(df_proportions)





