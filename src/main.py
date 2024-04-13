import math
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import IsolationForest as IF

from src.models.isolationForest.IsolationForest import IsolationForest


def main():
    # dataset preprocess
    # X_train_set, X_eval_set = preprocess_data("../Datasets/Kaggle/Financal_Data/BBCA.JK_weekly_new.csv")
    train_X, eval_X, train_y, eval_y = preprocess_data("../Datasets/Others/http.csv", "attack")

    # print(list(financal_data.columns.values)) #get columns name
    # print(financal_data['Date'].tolist()) #get all values from a column
    # print(X_train_set)
    # print(X_train_set.index.values)

    #df = df.sample(16000)

    build_start_time = time.time()
    iforest = IsolationForest(train_X, 30, 256)
    iforest.create()
    build_end_time = time.time()
    print(f"Forest build in: {build_end_time - build_start_time} seconds")

    score_start_time = time.time()
    scores = iforest.score(eval_X, 10)
    score_end_time = time.time()
    print(f"Score compute in: {score_end_time - score_start_time} seconds")

    eval_y = pd.DataFrame(eval_y, columns=['attack'])

    # scores[::-1].sort()
    threshold = iforest.percentile_threshold(scores, 0.4)
    print(f"percentile: {threshold}")

    predictions = iforest.predict_anomalies(scores, threshold)

    # add new columns with anomaly score and prediction for each observations
    eval_X['anomaly score'], eval_y['anomaly score'] = scores, scores
    eval_X['anomaly prediction'], eval_y['anomaly prediction'] = predictions, predictions
    print(eval_X.sort_values(by='anomaly score', ascending=False))
    print(eval_y.sort_values(by='anomaly score', ascending=False))

    print("-------------------------------------------------------------------")

    #clf = IF(n_estimators=300, max_samples=256, warm_start=True, random_state=0)
    #clf.fit(train_X)
    #scores2 = clf.decision_function(eval_x)
    # print(scores2)


def preprocess_data(file_path: str, prediction_column):
    dataframe = pd.read_csv(file_path)

    # drop column which contains categorical values and drop row with missing values
    X = dataframe[dataframe.select_dtypes(include=['float64', 'int64']).columns].dropna()

    if prediction_column:
        X, y = dataframe.drop(prediction_column, axis=1), dataframe[prediction_column]
        # split input data
        train_X, eval_X, train_y, eval_y = train_test_split(X, y, random_state=1)
        return train_X, eval_X, train_y, eval_y

    # split input data
    train_X, eval_X = train_test_split(X, random_state=1)
    return train_X, eval_X

if __name__ == "__main__":
    main()
