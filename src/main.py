import math

import pandas as pd
from sklearn.model_selection import train_test_split

#only for debugging
from sklearn.ensemble import IsolationForest as IF

from src.models.isolationForest.IsolationForest import IsolationForest


def main():
    # dataset preprocess
    #X_train_set, X_eval_set = preprocess_data("../Datasets/Kaggle/Financal_Data/BBCA.JK_weekly_new.csv")
    #X_train_set, X_eval_set = preprocess_data("../Datasets/Others/http.csv")
    # print(list(financal_data.columns.values)) #get columns name
    # print(financal_data['Date'].tolist()) #get all values from a column
    # print(X_train_set)
    # print(X_train_set.index.values)

    df = pd.read_csv("../Datasets/Others/http.csv")
    #N = 16_000
    #df = df.sample(N)  # grab random subset (too slow otherwise)

    X, y = df.drop('attack', axis=1), df['attack']

    iforest = IsolationForest(X, 30, 256)
    iforest.create()
    scores = iforest.score(X, 10)
    print(scores)

    print("-------------------------------------------------------------------")

    # clf = IF(n_estimators=300, max_samples=256, warm_start=True, random_state=0)
    # clf.fit(X)
    # scores2 = clf.decision_function(X)
    # print(scores2)

def preprocess_data(file_path):
    input_data = pd.read_csv(file_path)

    #drop column which contains categorical values
    processed_data = input_data[input_data.select_dtypes(include=['float64', 'int64']).columns]

    #drop row with missing values
    processed_data.dropna()

    #split input data
    train_data, eval_data = train_test_split(processed_data)
    return train_data, eval_data

def drop_date_columns(df):
    date_columns = df.select_dtypes(include=['datetime64']).columns
    df = df.drop(columns=date_columns)
    return df

if __name__ == "__main__":
    main()
