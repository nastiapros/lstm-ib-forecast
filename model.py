import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

# Генерация или загрузка данных
try:
    df = pd.read_csv('synthetic_incidents.csv')
except:
    print("Генерация синтетических данных...")
    def generate_data(n=240):
        time = np.arange(n)
        trend = time * 0.3
        seasonality = 10 * np.sin(2 * np.pi * time / 12)
        noise = np.random.normal(0, 5, n)
        incidents = np.maximum(20, 50 + trend + seasonality + noise)
        pd.DataFrame({'incidents': incidents}).to_csv('synthetic_incidents.csv', index=False)
    generate_data()
    df = pd.read_csv('synthetic_incidents.csv')

data = df['incidents'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, timesteps=12):
    X, y = [], []
    for i in range(len(dataset) - timesteps):
        X.append(dataset[i:i+timesteps])
        y.append(dataset[i+timesteps])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(12, 1)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[early_stop], verbose=1)

model.save_weights('lstm_ib_forecast_best.h5')
print("Модель обучена")

last_seq = scaled_data[-12:].reshape(1, 12, 1)
pred = model.predict(last_seq, verbose=0)
print(f"Прогноз на следующий квартал: {scaler.inverse_transform(pred)[0][0]:.1f} инцидентов")
