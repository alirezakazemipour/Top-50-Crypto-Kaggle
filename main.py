"""
References:
    - [Idea](https://neptune.ai/blog/predicting-stock-prices-using-machine-learning)
    - [Dataset](https://www.kaggle.com/odins0n/top-50-cryptocurrency-historical-prices)
"""
from utils import *
import wandb
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, GRU
from tensorflow.keras.regularizers import L2
import pandas as pd

if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)
    allow_gpu_growth()

    if os.path.exists("api_key.wandb"):
        with open("api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            os.environ["WANDB_MODE"] = "offline"
    else:
        raise FileNotFoundError("WandB API Token Not Found!")
    wandb.init(project="Cryptocurrencies' value prediction.")

    train_set, test_set = get_sets(args.crypto_name)

    window_range = "sma_" + str(args.window_size) + 'day'
    sma = train_set.rolling(args.window_size).mean()
    log_comparison_result_plot(train_set.assign(SMA=sma), "SMA", window_range, wandb)

    window_range = "ema_" + str(args.window_size) + 'day'
    ema = train_set.ewm(span=args.window_size, adjust=False).mean()
    log_comparison_result_plot(train_set.assign(EMA=ema), "EMA", window_range, wandb)

    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set.values)
    x_train, y_train = make_sequence(train_set, args.window_size, args.window_size)
    test_set = scaler.transform(test_set.values)
    x_test, y_test = make_sequence(test_set, args.window_size, args.window_size)

    model = Sequential(
        [
            Input(shape=x_train.shape[1:]),
            GRU(128, return_sequences=True),
            GRU(32, dropout=0.95),
            Dense(1)
        ]
    )
    model.summary()
    model.compile("adam", loss="mse")
    model.fit(x_train,
              y_train,
              batch_size=args.batch_size,
              epochs=args.epoch,
              callbacks=[wandb.keras.WandbCallback()],
              validation_split=0.15,
              verbose=1
              )

    y_pred = model.predict(x_test)
    test_df = pd.DataFrame({'Price': y_test})
    log_comparison_result_plot(test_df.assign(NN=y_pred), "NN", "NN", wandb)