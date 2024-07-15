import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import datetime

# Fetch data from Alpha Vantage
api_key = 'Z546U0RSBDK86YYE'
symbol = 'AAPL'

ts = TimeSeries(key=api_key, output_format='pandas')

# Fetching intraday data
data_15min, _ = ts.get_intraday(symbol=symbol, interval='15min', outputsize='full')
data_30min, _ = ts.get_intraday(symbol=symbol, interval='30min', outputsize='full')
data_1hour, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize='full')
data_daily, _ = ts.get_daily(symbol=symbol, outputsize='full')

# Function to aggregate intraday data into daily features
def aggregate_intraday(data, interval):
    data = data.reset_index()
    data['date'] = data['date'].dt.date
    aggregated = data.groupby('date').agg({
        '1. open': 'first',
        '2. high': 'max',
        '3. low': 'min',
        '4. close': 'last',
        '5. volume': 'sum'
    })
    aggregated.columns = [f'{col}_{interval}' for col in aggregated.columns]
    return aggregated

# Aggregate intraday data
data_15min_agg = aggregate_intraday(data_15min, '15min')
data_30min_agg = aggregate_intraday(data_30min, '30min')
data_1hour_agg = aggregate_intraday(data_1hour, '1hour')

# Combine aggregated intraday data with daily data using merge
data_daily = data_daily.reset_index()
data_daily['date'] = data_daily['date'].dt.date
data_combined = data_daily.merge(data_15min_agg, on='date', how='left')
data_combined = data_combined.merge(data_30min_agg, on='date', how='left', suffixes=('', '_30min'))
data_combined = data_combined.merge(data_1hour_agg, on='date', how='left', suffixes=('', '_1hour'))

# Drop rows with any missing values
data_combined = data_combined.dropna()

# Check the number of samples we have after combining the data
print(f'Number of samples after combining: {len(data_combined)}')

if len(data_combined) <= 1:
    raise ValueError("Not enough data samples after combining. Please check the data filtering criteria and ensure there is sufficient data.")

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

data_combined = data_combined.dropna()

# Check the number of samples we have after feature engineering
print(f'Number of samples after feature engineering: {len(data_combined)}')

if len(data_combined) <= 1:
    raise ValueError("Not enough data samples after feature engineering. Please check the data preprocessing steps and ensure there is sufficient data.")

# Define features and labels
features = [
    'previous_close', '1. open', 'ma5', 'ma10', 'ma20', 'vol_change', 'high_low_diff', 'open_close_diff',
    'ema5', 'ema20', 'momentum', 'volatility',
    '1. open_15min', '2. high_15min', '3. low_15min', '4. close_15min', '5. volume_15min',
    '1. open_30min', '2. high_30min', '3. low_30min', '4. close_30min', '5. volume_30min',
    '1. open_1hour', '2. high_1hour', '3. low_1hour', '4. close_1hour', '5. volume_1hour'
]
X = data_combined[features]
y = data_combined[['2. high', '3. low']]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets with more testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

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

# Output actual and predicted values for comparison
comparison = pd.DataFrame({
    'Actual_High': y_test.values[:, 0],
    'Actual_Low': y_test.values[:, 1],
    'Predicted_High_XGBoost': predictions_xgb[:, 0],
    'Predicted_Low_XGBoost': predictions_xgb[:, 1]
})

# Evaluate the performance
print(comparison)
print("XGBoost High RMSE:", np.sqrt(mean_squared_error(y_test.values[:, 0], predictions_xgb[:, 0])))
print("XGBoost Low RMSE:", np.sqrt(mean_squared_error(y_test.values[:, 1], predictions_xgb[:, 1])))
print("XGBoost High R2:", r2_score(y_test.values[:, 0], predictions_xgb[:, 0]))
print("XGBoost Low R2:", r2_score(y_test.values[:, 1], predictions_xgb[:, 1]))
