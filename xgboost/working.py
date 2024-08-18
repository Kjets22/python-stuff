
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Fetch data from Alpha Vantage
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

ts = TimeSeries(key=api_key, output_format='pandas')

# Define date range for 5 years
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Fetching intraday data (with compact size)
data_1min, _ = ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
data_5min, _ = ts.get_intraday(symbol=symbol, interval='5min', outputsize='compact')
data_15min, _ = ts.get_intraday(symbol=symbol, interval='15min', outputsize='compact')
data_30min, _ = ts.get_intraday(symbol=symbol, interval='30min', outputsize='compact')
data_1hour, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize='compact')

# Filter data to only include the last 5 years
data_1min = data_1min[data_1min.index >= start_date]
data_5min = data_5min[data_5min.index >= start_date]
data_15min = data_15min[data_15min.index >= start_date]
data_30min = data_30min[data_30min.index >= start_date]
data_1hour = data_1hour[data_1hour.index >= start_date]

# Concatenate all intraday data into a single DataFrame
data_combined = pd.concat([data_1min, data_5min, data_15min, data_30min, data_1hour])

# Reset index to have a single datetime index
data_combined = data_combined.reset_index()
data_combined = data_combined.rename(columns={'date': 'datetime'})

# Feature Engineering
data_combined['previous_close'] = data_combined['4. close'].shift(1)
data_combined['price_change'] = data_combined['4. close'] - data_combined['1. open']
data_combined['ma5'] = data_combined['4. close'].rolling(window=5, min_periods=1).mean()
data_combined['ma10'] = data_combined['4. close'].rolling(window=10, min_periods=1).mean()
data_combined['ma20'] = data_combined['4. close'].rolling(window=20, min_periods=1).mean()
data_combined['vol_change'] = data_combined['5. volume'].pct_change().fillna(0)
data_combined['high_low_diff'] = data_combined['2. high'] - data_combined['3. low']
data_combined['open_close_diff'] = data_combined['1. open'] - data_combined['4. close']

# Add more technical indicators
data_combined['ema5'] = data_combined['4. close'].ewm(span=5, adjust=False).mean()
data_combined['ema20'] = data_combined['4. close'].ewm(span=20, adjust=False).mean()
data_combined['momentum'] = data_combined['4. close'] - data_combined['4. close'].shift(4).fillna(0)
data_combined['volatility'] = data_combined['4. close'].rolling(window=5, min_periods=1).std()

# Add additional features
data_combined['roc'] = data_combined['4. close'].pct_change(periods=10)  # Rate of change
data_combined['ema12'] = data_combined['4. close'].ewm(span=12, adjust=False).mean()
data_combined['ema26'] = data_combined['4. close'].ewm(span=26, adjust=False).mean()
data_combined['macd'] = data_combined['ema12'] - data_combined['ema26']

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data_combined['rsi'] = calculate_rsi(data_combined['4. close'])

# Add lag features
lags = 30  # Increased the number of lags to capture more historical data
lag_columns = ['4. close', '2. high', '3. low', '5. volume']
lagged_data = pd.concat([data_combined[lag_columns].shift(i).add_suffix(f'_lag_{i}') for i in range(1, lags+1)], axis=1)
data_combined = pd.concat([data_combined, lagged_data], axis=1)

# Drop rows with any missing values after adding lag features
data_combined = data_combined.dropna()

# Check the number of samples we have after feature engineering
print(f'Number of samples after feature engineering: {len(data_combined)}')

if len(data_combined) <= 1:
    raise ValueError("Not enough data samples after feature engineering. Please check the data preprocessing steps and ensure there is sufficient data.")

# Replace infinite values and very large values
data_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
data_combined.fillna(0, inplace=True)

# Define features and labels
features = [
    'previous_close', '1. open', 'ma5', 'ma10', 'ma20', 'vol_change', 'high_low_diff', 'open_close_diff',
    'ema5', 'ema20', 'momentum', 'volatility', 'roc', 'macd', 'rsi'
] + [f'{col}_lag_{i}' for i in range(1, lags+1) for col in lag_columns]

X = data_combined[features]
y = data_combined[['2. high', '3. low']]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [500, 1000, 1500],     # Increase the number of trees
    'learning_rate': [0.01, 0.05, 0.1],    # Experiment with lower learning rates
    'max_depth': [10, 12, 15],             # Increase the tree depth
    'subsample': [0.8, 0.9, 1.0],          # Experiment with subsampling
    'colsample_bytree': [0.8, 0.9, 1.0],   # Control feature sampling per tree
    'alpha': [1, 10, 100],                 # L1 regularization
    'lambda': [10, 100, 1000]              # L2 regularization
}

# Use XGBRegressor with GPU support
xgb = XGBRegressor(random_state=42, tree_method='hist', device="cuda", n_jobs=1)

kf = KFold(n_splits=5)  # Increase the number of splits to 5
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_xgb = grid_search.best_estimator_

# Predict and evaluate XGBoost
predictions_xgb = best_xgb.predict(X_test)

# Calculate the actual accuracy metrics
mape_high = mean_absolute_percentage_error(y_test['2. high'], predictions_xgb[:, 0]) * 100
mape_low = mean_absolute_percentage_error(y_test['3. low'], predictions_xgb[:, 1]) * 100
mse_high = mean_squared_error(y_test['2. high'], predictions_xgb[:, 0])
mse_low = mean_squared_error(y_test['3. low'], predictions_xgb[:, 1])

# Print accuracy metrics
print(f"High Prediction MAPE: {mape_high:.2f}%")
print(f"Low Prediction MAPE: {mape_low:.2f}%")
print(f"High Prediction MSE: {mse_high:.2f}")
print(f"Low Prediction MSE: {mse_low:.2f}")

# Output actual and predicted values for comparison
comparison = pd.DataFrame({
    'Datetime': data_combined['datetime'][y_test.index],
    'Actual_High': y_test['2. high'],
    'Predicted_High': predictions_xgb[:, 0],
    'Actual_Low': y_test['3. low'],
    'Predicted_Low': predictions_xgb[:, 1]
})

# Plot the actual and predicted High and Low prices
plt.figure(figsize=(14, 5))
plt.scatter(comparison['Datetime'], comparison['Actual_High'], color='red', label='Actual High Price')
plt.scatter(comparison['Datetime'], comparison['Predicted_High'], color='blue', label='Predicted High Price')
plt.scatter(comparison['Datetime'], comparison['Actual_Low'], color='green', label='Actual Low Price')
plt.scatter(comparison['Datetime'], comparison['Predicted_Low'], color='orange', label='Predicted Low Price')

# Sort by Datetime to ensure lines are drawn in the correct order
comparison_sorted = comparison.sort_values(by='Datetime')

# Draw lines connecting the points by the closest dates
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Actual_High'], color='red', alpha=0.5, linestyle='-')
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Predicted_High'], color='blue', alpha=0.5, linestyle='-')
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Actual_Low'], color='green', alpha=0.5, linestyle='-')
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Predicted_Low'], color='orange', alpha=0.5, linestyle='-')

plt.title(f'{symbol} High and Low Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
