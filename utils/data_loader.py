import pandas as pd

def get_sets(dataset_path,
             valid_portion=0.15,
             test_portion=0.05
             ):
    df = pd.read_csv(dataset_path)
    data = df[["Price", "Open", "High", "Low", "Vol."]]

    train_portion = 1 - test_portion
    train_size = int(train_portion * len(data))
    train_data = data.loc[:train_size]
    test_df = data.loc[train_size:]

    train_portion = 1 - valid_portion
    train_size = int(train_portion * len(train_data))
    train_df = train_data.loc[:train_size]
    valid_df = train_data.loc[train_size:]

    return df, train_df, valid_df, test_df
