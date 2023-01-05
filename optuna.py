import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from tensorflow.keras.layers import Embedding
import numpy as np
import random
import optuna

#modelの名前
model_name = "./model1"

#optunaで最適化
def objective(trial):
    samples = np.loadtxt('./tfidf_unigram_train.csv', delimiter=',') #トレーニングデータ読み込み
    samples_label = samples[:, 0].astype(int)
    samples_data = samples[:, 1:]
    len_sequence = len(samples_data[0])           # 時系列の長さ
    samples_label = np.reshape(samples_label, (-1, 1, 1))
    samples_data = np.reshape(samples_data, (-1, len_sequence, 1))

    X, t = samples_data, samples_label
    input_dim = 1                # 入力データの次元数：実数値1個なので1を指定
    output_dim = 1               # 出力データの次元数：同上
    num_hidden_units = trial.suggest_int("num_hidden_units", 64, 128)      # 隠れ層のユニット数
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])              # ミニバッチサイズ
    num_of_training_epochs = trial.suggest_int("num_of_training_epochs", 50, 100) # 学習エポック数
    learning_rate = trial.suggest_uniform("learning_rate", 1e-4, 1e-2)        # 学習率
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
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
    # 学習
    history = model.fit(
        X, t,
        batch_size=batch_size,
        epochs=num_of_training_epochs,
        verbose = 0,
        validation_split=0.1
    )
    if min > history.history["val_loss"][-1]:
        print(min)
        model.save(model_name)
    return history.history["val_loss"][-1]

TRIAL_SIZE = 50
min = 1
study = optuna.create_study()
study.optimize(objective, n_trials=TRIAL_SIZE)
print(study.best_params)