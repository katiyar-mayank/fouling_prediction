import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from scipy.stats import zscore

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, Concatenate, GlobalAveragePooling1D, Dropout, LayerNormalization, Conv1D, MultiHeadAttention

# -------------------- Load & Clean --------------------
df = pd.read_csv('fouling.csv')
df.dropna(inplace=True)
df['ΔFouling'] = df['Fouling factor (m2 K/kW)'].diff()
df.dropna(inplace=True)

z = zscore(df['Fouling factor (m2 K/kW)'])
df = df[np.abs(z) < 3]

# -------------------- Scale --------------------
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
data = scaled_df.values

# -------------------- Config --------------------
input_len = 200
output_len = 50
num_features = data.shape[1]
use_residual_learning = False  # Optional toggle

# -------------------- Sequence Creation --------------------
X, y = [], []
for i in range(input_len, len(data) - output_len + 1):
    X.append(data[i - input_len:i])
    if use_residual_learning:
        # Predict difference
        delta_y = data[i:i + output_len, 1] - data[i - 1:i + output_len - 1, 1]
        y.append(delta_y)
    else:
        y.append(data[i:i + output_len, 1])
X = np.array(X)
y = np.array(y)[..., np.newaxis]  # shape: (N, output_len, 1)

# -------------------- Positional Encoding --------------------
class PositionalEncoding(Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_encoding = self.get_positional_encoding(seq_len, d_model)

    def get_positional_encoding(self, position, d_model):
        angles = np.arange(position)[:, np.newaxis] * 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# -------------------- Transformer Encoder --------------------
class TransformerEncoder(Layer):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.norm1 = LayerNormalization()
        self.dropout1 = Dropout(dropout)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation='relu'), Dense(d_model)])
        self.norm2 = LayerNormalization()
        self.dropout2 = Dropout(dropout)

    def call(self, x):
        attn_out = self.attn(x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_out))

# -------------------- Model --------------------
def create_model():
    inp = Input(shape=(input_len, num_features))

    # CNN Block
    x = Conv1D(64, 3, padding='causal', activation='relu')(inp)
    x = Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=2)(x)

    d_model, n_heads = 128, 4
    x = Dense(d_model)(x)
    x = PositionalEncoding(input_len, d_model)(x)

    # Transformer Layers
    for _ in range(2):
        x = TransformerEncoder(d_model, n_heads, ff_dim=256)(x)

    # Project to output sequence
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Conv1D(1, kernel_size=1)(x[:, -output_len:, :])  # output_len timesteps, 1-dim

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='mse', metrics=['mae', 'mape'])
    return model

# -------------------- Evaluation --------------------
def calculate_index_of_agreement(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean())) ** 2)
    return 1 - num / den

def evaluate_model(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return (
        mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
        calculate_index_of_agreement(y_true, y_pred),
        mean_absolute_percentage_error(y_true, y_pred)
    )


# -------------------- Train --------------------
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}")
    model = create_model()
    model.fit(X[train_idx], y[train_idx], batch_size=32, epochs=200, verbose=0)
    y_pred = model.predict(X[test_idx])
    scores = evaluate_model(y[test_idx], y_pred)
    print(f"MSE: {scores[0]:.6f}, MAE: {scores[1]:.6f}, R²: {scores[2]:.6f}, IOA: {scores[3]:.6f}, MAPE: {scores[4]:.6f}")

# -------------------- Forecast Plot --------------------
pred_scaled = model.predict(X[-1:])[0].squeeze()
true_scaled = y[-1].squeeze()

fouling_scaler = MinMaxScaler()
fouling_scaler.min_, fouling_scaler.scale_ = scaler.min_[1], scaler.scale_[1]
pred = fouling_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
true = fouling_scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()

final_r2 = r2_score(true, pred)
print(f"Final R² score on the forecasted sequence: {final_r2:.6f}")


plt.figure(figsize=(12, 4))
plt.plot(range(output_len), true, label='Actual')
plt.plot(range(output_len), pred, label='Predicted')
plt.title(f"Next {output_len}-step Forecast (Original Scale)")
plt.xlabel("Time Steps Ahead")
plt.ylabel("Fouling Factor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graph/Prediction.png")
plt.show()
