import math
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest as IF
from src.models.isolationForest.IsolationForest import IsolationForest


def main():
    # dataset preprocess
    #train_X, eval_X, _, _ = preprocess_data("../datasets/kaggle/financal_Data/BBCA.JK_weekly_new.csv", None)
    train_X, eval_X, train_y, eval_y = preprocess_data("../datasets/others/http.csv", "attack")
    #train_X, eval_X, train_y, eval_y = preprocess_data("../datasets/kaggle/ddos/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Label")

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

    #eval_y = pd.DataFrame(eval_y, columns=['attack'])

    # Percentile threshold
    # scores[::-1].sort()
    # threshold = iforest.percentile_threshold(scores, 0.75)
    # print(f"percentile: {threshold}")

    #TPR threshold
    threshold, tpr, _ = iforest.tpr_threshold(eval_y, scores, desired_tpr=.8)

    prediction_y = iforest.predict_anomalies(scores, threshold)

    print(f"TPR Threshold: {threshold}")
    print(f"TPR: {tpr}")

    compute_auc(eval_y, scores)

    # add new columns with anomaly score and prediction for each observations
    # eval_X['anomaly score'], eval_y['anomaly score'] = scores, scores
    # eval_X['anomaly prediction'], eval_y['anomaly prediction'] = prediction_y, prediction_y
    # print(eval_X.sort_values(by='anomaly score', ascending=False))
    # print(eval_y.sort_values(by='anomaly score', ascending=False))

    # confusion matrix
    # confusion_matrix = metrics.confusion_matrix(eval_y, prediction_y)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    # cm_display.plot()
    # plt.show()

    # ROC curve
    # fpr, tpr, _ = metrics.roc_curve(eval_y, prediction_y)
    # df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    # plt.plot(df['fpr'], df['tpr'], label='ROC Curve')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess', color='r')  # ligne de référence aléatoire
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend()
    # plt.show()


    print("-------------------------------------------------------------------")

    #clf = IF(n_estimators=300, max_samples=256, warm_start=True, random_state=0)
    #clf.fit(train_X)
    #scores2 = clf.decision_function(eval_x)
    # print(scores2)


def preprocess_data(file_path: str, prediction_column):
    df = pd.read_csv(file_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    # drop column which contains categorical values and drop row with missing values
    proc_df = df[df.select_dtypes(include=['float64', 'int64']).columns.tolist()].dropna()

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
    print("t")

if __name__ == "__main__":
    main()
