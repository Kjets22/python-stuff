
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Fetch data from Alpha Vantage
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

ts = TimeSeries(key=api_key, output_format='pandas')

# Define date range for 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

# Fetching intraday data (with compact size)
data_1min, _ = ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
data_5min, _ = ts.get_intraday(symbol=symbol, interval='5min', outputsize='compact')
data_15min, _ = ts.get_intraday(symbol=symbol, interval='15min', outputsize='compact')
data_30min, _ = ts.get_intraday(symbol=symbol, interval='30min', outputsize='compact')
data_1hour, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize='compact')

# Filter data to only include the last 3 years
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
lags = 5  # Adjust the number of lags as needed
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

# Use XGBRegressor with GPU support
xgb = XGBRegressor(random_state=42, tree_method='hist', device='cuda:0', n_jobs=-1)

kf = KFold(n_splits=5)  # Increase the number of splits to 5
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=2)
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

# Select 10 random test examples to print
sampled_comparison = comparison.sample(n=10, random_state=42)

# Print selected test examples
for index, row in sampled_comparison.iterrows():
    print(f"Example {index + 1}:")
    print(f"  Actual High: {row['Actual_High']}, Predicted High: {row['Predicted_High_XGBoost']}")
    print(f"  Actual Low: {row['Actual_Low']}, Predicted Low: {row['Predicted_Low_XGBoost']}")
    print()

# Calculate accuracy within ±10%
accuracy_high = np.mean(np.abs((comparison['Actual_High'] - comparison['Predicted_High_XGBoost']) / comparison['Actual_High']) <= 0.1) * 100
accuracy_low = np.mean(np.abs((comparison['Actual_Low'] - comparison['Predicted_Low_XGBoost']) / comparison['Actual_Low']) <= 0.1) * 100

# Print accuracy
print(f"High Prediction Accuracy: {accuracy_high:.2f}%")
print(f"Low Prediction Accuracy: {accuracy_low:.2f}%")

# Plot accuracy
plt.figure(figsize=(10, 6))
categories = ['High', 'Low']
accuracies = [accuracy_high, accuracy_low]

plt.bar(categories, accuracies, color=['blue', 'orange'])
plt.ylim(90, 100)
plt.ylabel('Accuracy (%)')
plt.title('Prediction Accuracy for High and Low Prices')
plt.show()
