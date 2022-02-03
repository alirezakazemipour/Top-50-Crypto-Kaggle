"""
References:
    - [Idea](https://neptune.ai/blog/predicting-stock-prices-using-machine-learning)
    - [Dataset](https://www.kaggle.com/odins0n/top-50-cryptocurrency-historical-prices)
"""
from utils import *
import wandb
import os
# from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout
import pandas as pd

if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)
    allow_gpu_growth()

    if os.path.exists("api_key.wandb"):
        with open("api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            if not args.online_wandb:
                os.environ["WANDB_MODE"] = "offline"
    else:
        raise FileNotFoundError("WandB API Token Not Found!")
    wandb.init(project="Cryptocurrencies' value prediction.")

    train_set, valid_set, test_set = get_sets(args.crypto_name)

    # window_range = "sma_" + str(args.window_size) + 'day'
    # sma = valid_set.rolling(args.window_size).mean()
    # log_comparison_result_plot(valid_set.assign(SMA=sma), "SMA", window_range, wandb)
    #
    # window_range = "ema_" + str(args.window_size) + 'day'
    # ema = valid_set.ewm(span=args.window_size, adjust=False).mean()
    # log_comparison_result_plot(valid_set.assign(EMA=ema), "EMA", window_range, wandb)

    # scaler = MinMaxScaler()
    # train_set = scaler.fit_transform(train_set.values)
    x_train, y_train = make_sequence(train_set, args.window_size, args.window_size)
    # valid_set = scaler.transform(valid_set.values)
    x_valid, y_valid = make_sequence(valid_set, args.window_size, args.window_size)
    # test_set = scaler.transform(test_set.values)
    x_test, y_test = make_sequence(test_set, args.window_size, args.window_size)

    model = Sequential(
        [
            Input(shape=x_train.shape[1:]),
            LSTM(128, return_sequences=True, ),
            LSTM(128),
            Dense(1)
        ]
    )
    model.summary()
    model.compile("adam", loss="mae")
    model.fit(x_train,
              y_train,
              batch_size=args.batch_size,
              epochs=args.epoch,
              callbacks=[wandb.keras.WandbCallback()],
              verbose=1,
              validation_data=(x_valid, y_valid),
              # shuffle=True
              )
    #
    y_pred = model.predict(x_test)
    tmp = pd.DataFrame({'Price': y_test})
    print(np.mean((y_pred - y_test) ** 2))
    log_comparison_result_plot(tmp.assign(NN=y_pred), "NN", "NN", wandb)
