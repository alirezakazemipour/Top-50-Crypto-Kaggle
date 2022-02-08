"""
References:
    - [Idea](https://neptune.ai/blog/predicting-stock-prices-using-machine-learning)
    - [Dataset](https://www.kaggle.com/odins0n/top-50-cryptocurrency-historical-prices)
"""
from utils import *
from models import get_model
import wandb
import os
import glob
from tqdm import tqdm

if __name__ == "__main__":
    args = get_args()
    allow_gpu_growth()

    if os.path.exists("api_key.wandb"):
        with open("api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            if not args.online_wandb:
                os.environ["WANDB_MODE"] = "offline"
    else:
        raise FileNotFoundError("WandB API Token Not Found!")
    wandb.init(project="Cryptocurrencies' value prediction.")

    cryptonames = ["Algorand", "Bitcoin", "BitTorrent", "Ethereum", "Tether", "Tron", "IOTA", "Binance_Coin", "Cosmos", "Dogecoin", "EOS"]
    path = "datasets/top-50-cryptocurrency-historical-prices"
    results = []
    models = {"cnn": {"config": dict(seq_len=args.window_size, in_dim=5),
                      "y_pred": None
                      },
              "transformer": {"config": dict(seq_len=args.window_size, in_dim=5, d_k=16, d_v=16, n_heads=3, ff_dim=16),
                              "y_pred": None
                              },
              "lstm": {"config": dict(seq_len=args.window_size, in_dim=5),
                       "y_pred": None
                       }
              }
    for name in tqdm(cryptonames):
        file = os.path.join(path, name + ".csv")
        print(f"==>{file.split(os.sep)[-1]}")
        dataset, train_set, valid_set, test_set = get_sets(file)

        x_train, y_train = make_sequence(train_set, args.window_size, args.window_size)
        x_valid, y_valid = make_sequence(valid_set, args.window_size, args.window_size)
        x_test, y_test = make_sequence(test_set, args.window_size, args.window_size)

        for model_name, d in models.items():
            set_random_seed(args.seed)
            model = get_model(model_name, **d["config"])
            # model.summary()
            model.compile(loss="huber", optimizer="adam", metrics=['mae', 'mape'])
            model.fit(x_train,
                      y_train,
                      batch_size=args.batch_size,
                      epochs=args.epoch,
                      callbacks=[wandb.keras.WandbCallback()],
                      verbose=0,
                      validation_data=(x_valid, y_valid),
                      )
            y_pred = model.predict(x_test)
            y_pred = np.squeeze(y_pred, -1)
            print(f"{model_name} test mse: {np.mean(np.square(y_pred - y_test)):.3f}")
            y_pred = inv_sequence(test_set, y_pred, args.window_size, args.window_size)
            tmp = len(dataset[["Price"]].values)
            train_size = int((1 - 0.05) * tmp)
            y_pred = np.append(dataset[["Price"]].values[:train_size + args.window_size], y_pred)
            models[model_name]["y_pred"] = y_pred

        y_true = dataset[["Price"]].values
        y_sma = dataset[["Price"]].rolling(args.window_size).mean().values
        y_ema = dataset[["Price"]].ewm(span=args.window_size, adjust=False).mean()
        results.append(
            (y_true,
             models["cnn"]["y_pred"],
             models["transformer"]["y_pred"],
             models["lstm"]["y_pred"],
             y_sma,
             y_ema,
             file.split(os.sep)[-1].split(".")[0]
             )
        )
    plot_results(results,
                 wandb
                 )