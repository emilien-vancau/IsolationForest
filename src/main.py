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
    train_X, eval_X, train_y, eval_y = preprocess_data("../datasets/others/http.csv", "attack", 100000)
    #train_X, eval_X, train_y, eval_y = preprocess_data("../datasets/kaggle/ddos/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Label")

    # print(list(financal_data.columns.values)) #get columns name
    # print(financal_data['Date'].tolist()) #get all values from a column
    # print(X_train_set)
    # print(X_train_set.index.values)

    #df = df.sample(16000)

    print("=== Begin training phase ===")
    iforest = IsolationForest(train_X, 30, 256)
    iforest.fit()
    print(f"Training time: {iforest.training_time} seconds")
    print("=== End training phase ===")
    print()

    print("=== Begin evaluation phase ===")
    scores = iforest.score(eval_X, 10)
    evaluation_end_time = time.time()
    print(f"Score compute in: {iforest.evaluation_time} seconds")
    print("=== End evaluation phase ===")

    #TPR threshold
    threshold, tpr1, fpr1 = iforest.tpr_threshold(eval_y, scores, desired_tpr=.8)

    prediction_y = iforest.predict_anomalies(scores, threshold)

    print(f"TPR Threshold found at: {threshold}")
    #print(f"TPR: {tpr1}")

    #print(f"AUC value: {compute_auc(eval_y, scores)}")


    # add new columns with anomaly score and prediction for each observations
    # eval_X['anomaly score'], eval_y['anomaly score'] = scores, scores
    # eval_X['anomaly prediction'], eval_y['anomaly prediction'] = prediction_y, prediction_y
    # print(eval_X.sort_values(by='anomaly score', ascending=False))
    # print(eval_y.sort_values(by='anomaly score', ascending=False))

    # confusion matrix
    fpr, tpr, _ = metrics.roc_curve(eval_y, prediction_y)
    print(f"AUC: {metrics.auc(fpr, tpr)}")
    confusion_matrix = metrics.confusion_matrix(eval_y, prediction_y)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.show()

    # ROC curve

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


def preprocess_data(file_path: str, prediction_column: str, sample_size):
    df = pd.read_csv(file_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    # randomly pre-sample dataset
    if sample_size:
        if df.shape[0] >= sample_size:
            df = df.sample(sample_size)
        else:
            print("sample size too high for dataframe")
            return

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
    df = pd.DataFrame({'score': anomalies_scores, 'label': anomalies_labels})
    true_anomalies = df[df['label'] == 1]
    df.sort_values(by='score', ascending=False)

    true_anomalies_cnt = true_anomalies.shape[0]
    true_normal_points_cnt = df.shape[0] - true_anomalies_cnt

    # get n firsts row in the ranked list
    df = df.head(true_anomalies_cnt)

    S = sum(df.index.values)


    #S = sum(rank for rank, label in enumerate(scores_labels, start=1) if label)

    AUC = (S - ((true_anomalies_cnt ** 2 + true_anomalies_cnt) / 2)) / (true_anomalies_cnt * true_normal_points_cnt)

    return AUC

if __name__ == "__main__":
    main()
