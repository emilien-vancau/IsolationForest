import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.isolationForest.IsolationForest import IsolationForest


def main():
    financal_train_set, financal_eval_set = preprocess_data("../Datasets/Kaggle/Financal_Data/BBCA.JK_weekly_new.csv")
    # print(list(financal_data.columns.values)) #get columns name
    # print(financal_data['Date'].tolist()) #get all values from a column

    iforest = IsolationForest(financal_train_set, 10, 10)
    iforest.init()
    print("test")


    #dataset preprocess

def preprocess_data(file_path):
    input_data = pd.read_csv(file_path)
    processed_input_data = input_data[input_data.select_dtypes(include=['float64', 'int64']).columns]
    train_data, eval_data = train_test_split(processed_input_data)
    return train_data, eval_data

def drop_date_columns(df):
    date_columns = df.select_dtypes(include=['datetime64']).columns
    df = df.drop(columns=date_columns)
    return df

if __name__ == "__main__":
    main()
