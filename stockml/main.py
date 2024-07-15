import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch data from Alpha Vantage
api_key = 'HCX1E7AHWL9RELPJ'
symbol = 'AAPL'

ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
data = data.sort_index(ascending=True) # Ensure data is in chronological order

# Filter data to include only the last three years
three_years_ago = datetime.now() - timedelta(days=3*365)
data = data[data.index >= three_years_ago.strftime('%Y-%m-%d')]

# Feature Engineering
data['previous_close'] = data['4. close'].shift(1)
data['price_change'] = data['4. close'] - data['1. open']
data['ma5'] = data['4. close'].rolling(window=5).mean()
data['ma10'] = data['4. close'].rolling(window=10).mean()
data['ma20'] = data['4. close'].rolling(window=20).mean()
data['vol_change'] = data['5. volume'].pct_change()
data['high_low_diff'] = data['2. high'] - data['3. low']
data['open_close_diff'] = data['1. open'] - data['4. close']

# Add more technical indicators
data['ema5'] = data['4. close'].ewm(span=5, adjust=False).mean()
data['ema20'] = data['4. close'].ewm(span=20, adjust=False).mean()
data['momentum'] = data['4. close'] - data['4. close'].shift(4)
data['volatility'] = data['4. close'].rolling(window=5).std()
data['rsi'] = 100 - (100 / (1 + data['4. close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                             data['4. close'].diff().apply(lambda x: abs(min(x, 0))).rolling(window=14).mean()))
data = data.dropna()

# Define features and labels
features = ['previous_close', '1. open', 'ma5', 'ma10', 'ma20', 'vol_change', 'high_low_diff', 'open_close_diff',
            'ema5', 'ema20', 'momentum', 'volatility', 'rsi']
X = data[features]
y = data['4. close']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9, 11],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 2, 3, 4, 5]
}

xgb = XGBRegressor(random_state=42)
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

# Best model from GridSearch
best_xgb = grid_search_xgb.best_estimator_

# Predict and evaluate XGBoost
predictions_xgb = best_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)
print(f"XGBoost - Mean Squared Error: {mse_xgb}")
print(f"XGBoost - R^2 Score: {r2_xgb}")

# Prepare data for LSTM
sequence_length = 20  # Use the past 20 days to predict the next day

# Scale data
scaler_lstm = StandardScaler()
data_scaled_lstm = scaler_lstm.fit_transform(data[features + ['4. close']])

# Create sequences
X_lstm = []
y_lstm = []
for i in range(sequence_length, len(data_scaled_lstm)):
    X_lstm.append(data_scaled_lstm[i-sequence_length:i, :-1])
    y_lstm.append(data_scaled_lstm[i, -1])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Split data (ensuring same test split)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Define LSTM model with more parameters
model = Sequential()
model.add(LSTM(1000, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(1000))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train_lstm, epochs=200, batch_size=64, verbose=2)

# Predict and evaluate LSTM
predictions_lstm = model.predict(X_test_lstm)

# Inverse transform the predictions and actual values
predictions_lstm = scaler_lstm.inverse_transform(
    np.hstack((X_test_lstm[:, -1, :], predictions_lstm))
)[:, -1]
y_test_lstm = scaler_lstm.inverse_transform(
    np.hstack((X_test_lstm[:, -1, :], y_test_lstm.reshape(-1, 1)))
)[:, -1]

mse_lstm = mean_squared_error(y_test_lstm, predictions_lstm)
r2_lstm = r2_score(y_test_lstm, predictions_lstm)
print(f"LSTM - Mean Squared Error: {mse_lstm}")
print(f"LSTM - R^2 Score: {r2_lstm}")

# Combine predictions (ensure the lengths are the same)
min_len = min(len(predictions_xgb), len(predictions_lstm))
predictions_xgb = predictions_xgb[:min_len]
predictions_lstm = predictions_lstm[:min_len]
y_test_combined = y_test[:min_len]

final_predictions = (predictions_xgb + predictions_lstm) / 2

# Evaluate combined model
mse_ensemble = mean_squared_error(y_test_combined, final_predictions)
r2_ensemble = r2_score(y_test_combined, final_predictions)
print(f"Ensemble Model - Mean Squared Error: {mse_ensemble}")
print(f"Ensemble Model - R^2 Score: {r2_ensemble}")

# Detailed daily results
comparison = pd.DataFrame({
    'Date': data.index[-len(y_test_combined):],
    'Actual_Close': y_test_combined,
    'Predicted_Close_XGBoost': predictions_xgb,
    'Predicted_Close_LSTM': predictions_lstm,
    'Predicted_Close_Ensemble': final_predictions
})
for index, row in comparison.iterrows():
    print(f"Date: {row['Date']}, Actual Close: {row['Actual_Close']:.2f}, Predicted Close XGBoost: {row['Predicted_Close_XGBoost']:.2f}, Predicted Close LSTM: {row['Predicted_Close_LSTM']:.2f}, Predicted Close Ensemble: {row['Predicted_Close_Ensemble']:.2f}")
