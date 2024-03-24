import pandas as pd
from sklearn.model_selection import train_test_split

#only for debugging
from sklearn.ensemble import IsolationForest as IF

from src.models.isolationForest.IsolationForest import IsolationForest


def main():
    # dataset preprocess
    financal_train_set, financal_eval_set = preprocess_data("../Datasets/Kaggle/Financal_Data/BBCA.JK_weekly_new.csv")
    # print(list(financal_data.columns.values)) #get columns name
    # print(financal_data['Date'].tolist()) #get all values from a column
    # print(financal_train_set)
    # print(financal_train_set.index.values)

    iforest = IsolationForest(financal_train_set, 10, 256)
    iforest.create()
    scores = iforest.score(financal_train_set, 10)
    print(scores)

    clf = IF(n_estimators=10, max_samples=256, warm_start=True, random_state=0)
    clf.fit(financal_train_set)
    scores2 = clf.score_samples(financal_train_set)
    print(scores2)

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
