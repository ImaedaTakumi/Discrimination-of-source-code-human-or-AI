import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.keras.layers import Embedding
from keras.optimizers import Adam
import numpy as np
import statistics

input_dim = 1                # 入力データの次元数：実数値1個なので1を指定
output_dim = 1               # 出力データの次元数：同上
num_hidden_units = 80      # 隠れ層のユニット数
batch_size = 128           # ミニバッチサイズ
num_of_training_epochs = 21 # 学習エポック数
learning_rate = 0.0002481117814739368        # 学習率
samples = np.loadtxt('./tfidf_unigram_train.csv', delimiter=',') #トレーニングデータ読み込み
samples_label = samples[:, 0].astype(int)
samples_data = samples[:, 1:]
len_sequence = len(samples_data[0])           # 時系列の長さ
samples_label = np.reshape(samples_label, (-1, 1, 1))
samples_data = np.reshape(samples_data, (-1, len_sequence, 1))

X, t = samples_data, samples_label

model = Sequential()
model.add(Embedding(
    input_dim=len_sequence, # 入力として取り得るカテゴリ数（パディングの0を含む）
    output_dim=input_dim,                # 出力ユニット数（本来の特徴量の次元数）
    mask_zero=True))                     # 0をパディング用に特別扱いする  
model.add(LSTM(
    num_hidden_units,
    input_shape=(len_sequence, input_dim),
    return_sequences=False))
model.add(Dense(output_dim))
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=learning_rate))
# 学習
model.fit(
    X, t,
    batch_size=batch_size,
    epochs=num_of_training_epochs,
    validation_split=0.1
)

#モデルの保存(10回行うが1回毎に取ること)
model.save("unigram_model")

# 予測
# (サンプル, 時刻, 特徴量の次元) の3次元の入力を与える。
t_test = np.loadtxt('./tfidf_unigram_test.csv', delimiter=',')
test_label = t_test[:, 0].astype(int)
test_data = t_test[:, 1:]
test = test_data.reshape((len(test_label), len_sequence, 1))
result = model.predict(test)
val_acc = model.evaluate(test)

#予測結果見る
import statistics

print(result)
result_sort = [i[0] for i in result]
print(sorted(set(result_sort), key=result_sort.index))
print(f"重複なし要素数:{len(result_sort)}")

#0.5より上なら1、下なら0と無理やりラベル付け
median = statistics.median(result_sort)
label_predict_list = []
test_label_list = test_label.tolist()
for i in result:
  if i[0] > 0.5:
    label_predict_list.append(1)
  elif i[0] < 0.5:
    label_predict_list.append(0)
  else:
    label_predict_list.append(-1)

print(test_label_list)
print(label_predict_list)

t = 0
f = 0
for i in range(len(label_predict_list)):
  if label_predict_list[i] == test_label_list[i]:
    t += 1
  else:
    f += 1

print(f"精度{t/len(label_predict_list)}")