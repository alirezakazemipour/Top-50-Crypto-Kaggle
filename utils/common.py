import numpy as np
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt


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


def log_comparison_result_plot(df, method, window_range, wandb):
    ax = df[['Price', method]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.axis('tight')
    plt.ylabel('Stock Price ($)')
    plt.legend(["Price", window_range])
    wandb.log({method: wandb.Image(ax)})


def make_sequence(data, n, offset):
    data = data.values
    x, y = [], []

    for i in range(offset, len(data)):
        x.append(data[i - n:i] / data[i - n] - 1)
        y.append(data[i][0] / data[i - n][0] - 1)

    return np.stack(x), np.hstack(y)
