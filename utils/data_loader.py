import pandas as pd
import matplotlib.pyplot as plt


def get_sets(crypto_name,
             valid_portion=0.15,
             test_portion=0.05
             ):
    dataset_path = "datasets/top-50-cryptocurrency-historical-prices/" + crypto_name + ".csv"
    df = pd.read_csv(dataset_path)
    data = df[["Price"]]
    # abs_base = data.iloc[0]
    # data = (data / abs_base) - 1

    train_portion = 1 - test_portion
    train_size = int(train_portion * len(data))
    train_data = data.loc[:train_size]
    test_df = data.loc[train_size:]

    train_portion = 1 - valid_portion
    train_size = int(train_portion * len(train_data))
    train_df = train_data.loc[:train_size]
    valid_df = train_data.loc[train_size:]

    return train_df, valid_df, test_df
