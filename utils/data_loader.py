import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_sets(crypto_name,
             valid_portion=0.1,
             test_portion=0.1
             ):
    dataset_path = "datasets/top-50-cryptocurrency-historical-prices/" + crypto_name + ".csv"
    df = pd.read_csv(dataset_path)
    data = df[["Price", "Low", "High", "Open"]]

    train_portion = 1 - test_portion
    train_size = int(train_portion * len(data))
    train_data = data.loc[:train_size]
    test_df = data.loc[train_size:]

    train_portion = 1 - valid_portion
    train_size = int(train_portion * len(train_data))
    train_df = train_data.loc[:train_size]
    valid_df = train_data.loc[train_size:]

    return train_df, valid_df, test_df
