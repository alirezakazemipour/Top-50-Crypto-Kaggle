import pandas as pd


def get_sets(crypto_name,
             test_portion=0.5
             ):
    dataset_path = "datasets/top-50-cryptocurrency-historical-prices/" + crypto_name + ".csv"
    df = pd.read_csv(dataset_path)
    data = df[["Price"]]

    train_portion = 1 - test_portion
    train_size = train_portion * len(data)
    train_df = data.loc[train_size:]
    test_df = data.loc[:train_size]

    return train_df, test_df
