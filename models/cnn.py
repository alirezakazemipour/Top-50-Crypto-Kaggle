from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, MaxPooling1D


def create_cnn(seq_len, in_dim):
    model = Sequential(
        [
            Input(shape=(seq_len, in_dim)),
            Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1)
        ]
    )
    return model
