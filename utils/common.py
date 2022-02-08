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


def log_comparison_result_plot(y_true, y_pred, method, wandb):
    fig = plt.figure()
    x = range(len(y_true))
    plt.plot(x, y_true)
    x = range(len(y_pred))
    plt.plot(x, y_pred)
    plt.grid(False)
    plt.axis('tight')
    plt.ylabel('Stock Price ($)')
    plt.legend(["Price", "Predicted"])
    plt.grid()
    wandb.log({method: wandb.Image(fig)})


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
