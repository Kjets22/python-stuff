import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch data from Alpha Vantage
api_key = 'Z546U0RSBDK86YYE'
symbol = 'AAPL'

ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
data = data.sort_index(ascending=True)  # Ensure data is in chronological order

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

data = data.dropna()

# Define features and labels
features = ['previous_close', '1. open', 'ma5', 'ma10', 'ma20', 'vol_change', 'high_low_diff', 'open_close_diff', 'ema5', 'ema20', 'momentum', 'volatility']
X = data[features]
y = data['4. close']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_xgb = grid_search.best_estimator_

# Predict and evaluate XGBoost
predictions_xgb = best_xgb.predict(X_test)

# Prepare data for LSTM
sequence_length = 10  # Use the past 10 days to predict the next day

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

# Define LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, verbose=2)

# Predict and evaluate LSTM
predictions_lstm = model.predict(X_test_lstm)

# Inverse transform the predictions and actual values
predictions_lstm = scaler_lstm.inverse_transform(
    np.hstack((X_test_lstm[:, -1, :], predictions_lstm))
)[:, -1]
y_test_lstm = scaler_lstm.inverse_transform(
    np.hstack((X_test_lstm[:, -1, :], y_test_lstm.reshape(-1, 1)))
)[:, -1]

# Combine predictions (ensure the lengths are the same)
min_len = min(len(predictions_xgb), len(predictions_lstm))
predictions_xgb = predictions_xgb[:min_len]
predictions_lstm = predictions_lstm[:min_len]
y_test_combined = y_test[:min_len]

final_predictions = (predictions_xgb + predictions_lstm) / 2

# Output actual and predicted values for comparison
comparison = pd.DataFrame({
    'Actual_Close': y_test_combined,
    'Predicted_Close_XGBoost': predictions_xgb,
    'Predicted_Close_LSTM': predictions_lstm,
    'Predicted_Close_Ensemble': final_predictions
})
print(comparison[['Actual_Close', 'Predicted_Close_XGBoost', 'Predicted_Close_LSTM', 'Predicted_Close_Ensemble']])
