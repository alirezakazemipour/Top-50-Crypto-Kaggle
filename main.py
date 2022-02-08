"""
References:
    - [Idea](https://neptune.ai/blog/predicting-stock-prices-using-machine-learning)
    - [Dataset](https://www.kaggle.com/odins0n/top-50-cryptocurrency-historical-prices)
"""
import matplotlib.pyplot as plt
import numpy as np

from utils import *
import wandb
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Layer, Conv1D, LayerNormalization, Dropout, Concatenate, \
    GlobalAveragePooling1D
import tensorflow as tf
import glob


class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, inputs, *args, **kwargs):
        x = tf.math.reduce_mean(inputs, axis=-1)  # Convert (batch, seq_len, -1) to (batch, seq_len)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)  # (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # (batch, seq_len, 2)


class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')
        self.key = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')
        self.value = Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out


class MultiAttention(Layer):
    def __init__(self, in_dim, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
        self.linear = Dense(self.in_dim + 2, input_shape=input_shape, kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear


class TransformerEncoder(Layer):
    def __init__(self, in_dim, d_k, d_v, n_heads, ff_dim, dropout=0., **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.in_dim, self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        self.ff_conv1D_2 = Conv1D(filters=self.in_dim + 2,
                                  kernel_size=1)  # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer


def create_model(in_dim, seq_len, d_k, d_v, n_heads, ff_dim):
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    attn_layer1 = TransformerEncoder(in_dim, d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(in_dim, d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(in_dim, d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = Input(shape=(seq_len, in_dim))
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=in_seq, outputs=out)
    return model


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

    files = glob.glob("datasets/top-50-cryptocurrency-historical-prices/*.csv")
    files.sort()
    for file in files:
        print(f"==>{file.split(os.sep)[-1]}")
        set_random_seed(args.seed)
        dataset, train_set, valid_set, test_set = get_sets(file)

        y_true = dataset[["Price"]].values
        y_sma = dataset[["Price"]].rolling(args.window_size).mean().values
        y_ema = dataset[["Price"]].ewm(span=args.window_size, adjust=False).mean()
        log_comparison_result_plot(y_true, y_sma, "SMA", wandb)
        log_comparison_result_plot(y_true, y_ema, "EMA", wandb)

        x_train, y_train = make_sequence(train_set, args.window_size, args.window_size)
        x_valid, y_valid = make_sequence(valid_set, args.window_size, args.window_size)
        x_test, y_test = make_sequence(test_set, args.window_size, args.window_size)

        model = create_model(x_train.shape[-1], args.window_size, 16, 16, 25, 16)
        model.summary()
        model.compile(loss="huber", optimizer=tf.keras.optimizers.Adam()
                      , metrics=['mae', 'mape'])
        model.fit(x_train,
                  y_train,
                  batch_size=args.batch_size,
                  epochs=args.epoch,
                  callbacks=[wandb.keras.WandbCallback()],
                  verbose=2,
                  validation_data=(x_valid, y_valid),
                  )
        y_pred = model.predict(x_test)
        y_pred = np.squeeze(y_pred, -1)
        print(np.mean((y_pred - y_test) ** 2))
        y_true = dataset["Price"]
        y_pred = inv_sequence(test_set, y_pred, args.window_size, args.window_size)
        y_train = model.predict(x_train)
        y_train = inv_sequence(train_set, y_train, args.window_size, args.window_size)
        y_valid = model.predict(x_valid)
        y_valid = inv_sequence(valid_set, y_valid, args.window_size, args.window_size)
        y_pred = np.concatenate([y_train, y_valid, y_pred])
        log_comparison_result_plot(y_true, y_pred, "NN", wandb)
