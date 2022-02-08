import numpy as np
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def allow_gpu_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plot_results(results, wandb):
    fig, axes = plt.subplots(nrows=5, ncols=len(results), figsize=(5, 10))
    for i, result in enumerate(results):
        y_true, y_cnn, y_transformer, y_lstm, y_sma, y_ema, coin = result
        x = range(len(y_true))
        axes[0, i].set_title(coin + "_" + "CNN")
        axes[0, i].plot(x, y_true)
        axes[0, i].plot(x, y_cnn)
        axes[0, i].set_ylabel('Stock Price ($)')
        axes[0, i].legend(["Price", "Predicted"])
        axes[1, i].set_title(coin + "_" + "Transformer")
        axes[1, i].plot(x, y_true)
        axes[1, i].plot(x, y_transformer)
        axes[1, i].set_ylabel('Stock Price ($)')
        axes[1, i].legend(["Price", "Predicted"])
        axes[2, i].set_title(coin + "_" + "LSTM")
        axes[2, i].plot(x, y_true)
        axes[2, i].plot(x, y_lstm)
        axes[2, i].set_ylabel('Stock Price ($)')
        axes[2, i].legend(["Price", "Predicted"])
        axes[3, i].set_title(coin + "_" + "SMA")
        axes[3, i].plot(x, y_true)
        axes[3, i].plot(x, y_sma)
        axes[3, i].set_ylabel('Stock Price ($)')
        axes[3, i].legend(["Price", "Predicted"])
        axes[4, i].set_title(coin + "_" + "EMA")
        axes[4, i].plot(x, y_true)
        axes[4, i].plot(x, y_ema)
        axes[4, i].set_ylabel('Stock Price ($)')
        axes[4, i].legend(["Price", "Predicted"])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()
    wandb.log({"result": wandb.Image(fig)})


def make_sequence(data, n, offset, eps=1e-6):
    # https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning/
    data = data.values
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i - n:i] / (data[i - n] + eps) - 1)
        y.append(data[i][0] / (data[i - n][0] + eps) - 1)

    return np.stack(x), np.hstack(y)


def inv_sequence(data, y_pred, n, offset, eps=1e-6):
    # https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning/
    data = data.values
    x, y = [], []
    for i in range(offset, len(data)):
        y.append((data[i - n][0] + eps) * (y_pred[i - offset] + 1))

    return np.hstack(y)
