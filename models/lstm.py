from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense


def create_lstm(seq_len, in_dim):
    model = Sequential(
        [
            Input(shape=(seq_len, in_dim)),
            LSTM(128, return_sequences=True, ),
            LSTM(128),
            Dense(1)
        ]
    )
    return model
