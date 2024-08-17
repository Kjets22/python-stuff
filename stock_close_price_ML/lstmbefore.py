
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure TensorFlow uses the GPU

import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Verify if TensorFlow is using GPU
print(tf.config.list_physical_devices('GPU'))

# Function to calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Fetch data from Alpha Vantage
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

ts = TimeSeries(key=api_key, output_format='pandas')

# Define date range for 5 years
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Fetching intraday data
intervals = ['1min', '5min', '15min', '30min', '60min']
dataframes = []

for interval in intervals:
    data, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
    data = data.sort_index()
    dataframes.append(data)

# Preprocess data
def preprocess_data(data):
    data = data[['2. high', '3. low', '4. close', '5. volume']]
    data = data.rename(columns={'2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
    data['High'] = data['High'].astype(float)
    data['Low'] = data['Low'].astype(float)
    data['Close'] = data['Close'].astype(float)
    data['Volume'] = data['Volume'].astype(float)
    return data

preprocessed_dataframes = [preprocess_data(df) for df in dataframes]

# Add technical indicators
def add_technical_indicators(data):
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['RSI'] = calculate_rsi(data['Close'])
    data['Upper Band'] = data['MA20'] + (data['Volatility'] * 2)
    data['Lower Band'] = data['MA20'] - (data['Volatility'] * 2)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['ROC'] = data['Close'].pct_change(periods=10)
    return data

technical_dataframes = [add_technical_indicators(df) for df in preprocessed_dataframes]

# Drop rows with NaN values
cleaned_dataframes = [df.dropna() for df in technical_dataframes]

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataframes = [scaler.fit_transform(df) for df in cleaned_dataframes]

# Create a dataset with look_back periods
def create_dataset(dataset, look_back=100):
    X, Y_high, Y_low = [], [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), :])
        Y_high.append(dataset[i + look_back, 0])  # High
        Y_low.append(dataset[i + look_back, 1])   # Low
    return np.array(X), np.array(Y_high), np.array(Y_low)

look_back = 100
dataset_splits = [create_dataset(data, look_back) for data in scaled_dataframes]

# Check dimensions of created datasets
for i, split in enumerate(dataset_splits):
    print(f"Interval: {intervals[i]}, X shape: {split[0].shape}, Y_high shape: {split[1].shape}, Y_low shape: {split[2].shape}")

# Define LSTM model for each interval
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=150, return_sequences=False))
    model.add(Dropout(0.3))
    return model

# Build and train individual LSTM models
lstm_models = []
lstm_features_train = []
lstm_features_test = []

# Padding sequences to match lengths
max_length = max([split[0].shape[0] for split in dataset_splits])

for split in dataset_splits:
    X, Y_high, Y_low = split
    if X.ndim == 3 and len(X) > 0:  # Ensure X has 3 dimensions and is not empty
        X = np.pad(X, ((0, max_length - X.shape[0]), (0, 0), (0, 0)), 'constant')
        Y_high = np.pad(Y_high, (0, max_length - Y_high.shape[0]), 'constant')
        Y_low = np.pad(Y_low, (0, max_length - Y_low.shape[0]), 'constant')
        
        X_train, X_test, Y_train_high, Y_test_high, Y_train_low, Y_test_low = train_test_split(
            X, Y_high, Y_low, test_size=0.4, random_state=42)
        
        model = build_lstm_model((look_back, X.shape[2]))
        model.add(Dense(units=2))  # Predict both high and low
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, np.column_stack((Y_train_high, Y_train_low)), epochs=200, batch_size=32, validation_data=(X_test, np.column_stack((Y_test_high, Y_test_low))))
        
        lstm_models.append(model)
        lstm_features_train.append(model.predict(X_train))
        lstm_features_test.append(model.predict(X_test))

if lstm_features_train:
    # Combine features from all LSTM models
    combined_features_train = np.concatenate(lstm_features_train, axis=1)
    combined_features_test = np.concatenate(lstm_features_test, axis=1)

    # Train final model on combined features
    final_model = Sequential()
    final_model.add(Dense(units=512, activation='relu', input_shape=(combined_features_train.shape[1],)))
    final_model.add(Dropout(0.3))
    final_model.add(Dense(units=256, activation='relu'))
    final_model.add(Dropout(0.3))
    final_model.add(Dense(units=128, activation='relu'))
    final_model.add(Dense(units=2))  # Predict both high and low
    final_model.compile(optimizer='adam', loss='mean_squared_error')
    final_model.fit(combined_features_train, np.column_stack((Y_train_high, Y_train_low)), epochs=200, batch_size=32, validation_data=(combined_features_test, np.column_stack((Y_test_high, Y_test_low))))

    # Predicting the test set results
    predicted_stock_price = final_model.predict(combined_features_test)
    predicted_high = scaler.inverse_transform(np.concatenate((predicted_stock_price[:, 0].reshape(-1, 1), np.zeros((predicted_stock_price.shape[0], scaled_dataframes[0].shape[1] - 1))), axis=1))[:,0]
    predicted_low = scaler.inverse_transform(np.concatenate((predicted_stock_price[:, 1].reshape(-1, 1), np.zeros((predicted_stock_price.shape[0], scaled_dataframes[0].shape[1] - 1))), axis=1))[:,0]

    # Inverse transform the actual values
    real_high = scaler.inverse_transform(np.concatenate((Y_test_high.reshape(-1, 1), np.zeros((Y_test_high.shape[0], scaled_dataframes[0].shape[1] - 1))), axis=1))[:,0]
    real_low = scaler.inverse_transform(np.concatenate((Y_test_low.reshape(-1, 1), np.zeros((Y_test_low.shape[0], scaled_dataframes[0].shape[1] - 1))), axis=1))[:,0]

    # Calculate the root mean squared error
    rmse_high = np.sqrt(mean_squared_error(real_high, predicted_high))
    rmse_low = np.sqrt(mean_squared_error(real_low, predicted_low))
    print(f'Root Mean Squared Error (High): {rmse_high}')
    print(f'Root Mean Squared Error (Low): {rmse_low}')

    # Visualizing the results
    plt.figure(figsize=(14, 5))
    plt.plot(real_high, color='red', label='Actual High Price')
    plt.plot(predicted_high, color='blue', label='Predicted High Price')
    plt.title(f'{symbol} High Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(real_low, color='red', label='Actual Low Price')
    plt.plot(predicted_low, color='blue', label='Predicted Low Price')
    plt.title(f'{symbol} Low Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
else:
    print("No valid LSTM features were extracted. Check the dataset creation and preprocessing steps.")
