import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, Concatenate, GlobalAveragePooling1D, Dropout, LayerNormalization, Conv1D

# -------------------- Load and Scale Data --------------------
df = pd.read_csv('fouling.csv')
df.dropna(inplace=True)
data = df[['Time (hr)', 'Fouling factor (m2 K/kW)']].values

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
data = scaled_df.values
seq_len = 28
num_features = data.shape[1]

# -------------------- Data Preparation --------------------
X, y = [], []
for i in range(seq_len, len(data)):
    X.append(data[i-seq_len:i])
    y.append(data[i, 1])  # Index 1 = 'Fouling factor'
X, y = np.array(X), np.array(y)

# -------------------- Custom Layers --------------------
class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(self.seq_len,),
                                              initializer='uniform',
                                              trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(self.seq_len,),
                                           initializer='uniform',
                                           trainable=True)
        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(self.seq_len,),
                                                initializer='uniform',
                                                trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(self.seq_len,),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        x_mean = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = tf.expand_dims(self.weights_linear * x_mean + self.bias_linear, axis=-1)
        time_periodic = tf.expand_dims(tf.math.sin(x_mean * self.weights_periodic + self.bias_periodic), axis=-1)
        return tf.concat([time_linear, time_periodic], axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k)
        self.key = Dense(self.d_k)
        self.value = Dense(self.d_v)

    def call(self, inputs):
        q, k, v = self.query(inputs[0]), self.key(inputs[1]), self.value(inputs[2])
        attn_weights = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        return tf.matmul(attn_weights, v)


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = [SingleAttention(d_k, d_v) for _ in range(n_heads)]

    def build(self, input_shape):
        self.linear = Dense(input_shape[0][-1])

    def call(self, inputs):
        attn = [head(inputs) for head in self.attn_heads]
        return self.linear(tf.concat(attn, axis=-1))


class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.attn = MultiAttention(d_k, d_v, n_heads)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.conv1 = Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        self.conv2 = Conv1D(filters=None, kernel_size=1)  # will define in build()
        self.dropout2 = Dropout(dropout)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.conv2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)

    def call(self, inputs):
        x, y, z = inputs
        attn_out = self.attn((x, y, z))
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.conv2(self.conv1(x))
        return self.norm2(x + self.dropout2(ff))

# -------------------- Model Creation --------------------
def create_model():
    time_embedding = Time2Vector(seq_len)
    in_seq = Input(shape=(seq_len, num_features))
    x = Concatenate(axis=-1)([in_seq, time_embedding(in_seq)])
    for _ in range(3):
        x = TransformerEncoder(d_k=256, d_v=256, n_heads=12, ff_dim=256)([x, x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1)(x)

    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    return model

# -------------------- Evaluation Functions --------------------
def calculate_index_of_agreement(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    return 1 - num / den

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    ioa = calculate_index_of_agreement(y_true, y_pred)
    return mse, mae, r2, ioa, mape

def print_evaluation_results_extended(mse, mae, r2, ioa, mape):
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R^2: {r2:.6f}, IOA: {ioa:.6f}, MAPE: {mape:.6f}")

# -------------------- K-Fold Training --------------------
n_splits = 2
batch_size = 128
epochs = 200

kf = KFold(n_splits=n_splits, shuffle=False)
mse_scores, mae_scores, r2_scores, ioa_scores, mape_scores = [], [], [], [], []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = create_model()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    y_pred = model.predict(X_test).flatten()
    mse, mae, r2, ioa, mape = evaluate_model(y_test, y_pred)

    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    ioa_scores.append(ioa)
    mape_scores.append(mape)

    print_evaluation_results_extended(mse, mae, r2, ioa, mape)

# -------------------- Final Scores --------------------
print("\n=== Final Cross-Validation Results ===")
print_evaluation_results_extended(np.mean(mse_scores), np.mean(mae_scores), np.mean(r2_scores), np.mean(ioa_scores), np.mean(mape_scores))

# -------------------- Final Prediction --------------------
y_pred = model.predict(X).flatten()
mse, mae, r2, ioa, mape = evaluate_model(y, y_pred)

print("\n=== Final Model on Full Dataset ===")
print_evaluation_results_extended(mse, mae, r2, ioa, mape)

# -------------------- Plot Prediction --------------------
plt.figure(figsize=(12, 5))
plt.plot(range(len(y_pred)),y, label='Actual', linewidth=2)
plt.plot( range(len(y_pred)),y_pred, label='Predicted', linewidth=1.5)
plt.title("Final Model Prediction on Entire Dataset")
plt.xlabel("Time Step")
plt.ylabel("Fouling Factor (scaled)")
plt.legend()
plt.tight_layout()
plt.savefig(f'graph/Prediction_Entire_Dataset.png')
plt.show()
