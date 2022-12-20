import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import numpy as np
import random

input_dim = 1                # 入力データの次元数：実数値1個なので1を指定
output_dim = 1               # 出力データの次元数：同上
num_hidden_units = 128      # 隠れ層のユニット数
batch_size = 20            # ミニバッチサイズ
num_of_training_epochs = 40 # 学習エポック数
learning_rate = 0.001         # 学習率


samples = np.loadtxt('tfidf_1gram.csv', delimiter=',')
samples_label = samples[:, 0].astype(int)
samples_data = samples[:, 1:]
num_training_samples = len(samples)  # 学習データのサンプル数
len_sequence = len(samples_data[0])        # 時系列の長さ
samples_label = np.reshape(samples_label, (-1, 1, 1))
samples_data = np.reshape(samples_data, (-1, len_sequence, 1))

X, t = samples_data, samples_label

# モデル構築
model = Sequential()
model.add(LSTM(
    num_hidden_units,
    input_shape=(len_sequence, input_dim),
    return_sequences=False))
model.add(Dense(output_dim))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
model.summary()

# 学習
model.fit(
    X, t,
    batch_size=batch_size,
    epochs=num_of_training_epochs,
    validation_split=0.1
)

# 予測
# (サンプル, 時刻, 特徴量の次元) の3次元の入力を与える。
samples = np.loadtxt('tfidf_1gram_test.csv', delimiter=',')

t_test = np.loadtxt('tfidf_1gram_test.csv', delimiter=',')
test_data = t_test[:, 1:]
num_test_samples = len(test_data)
test = test_data.reshape((4, len_sequence, 1))

print(model.predict(test))