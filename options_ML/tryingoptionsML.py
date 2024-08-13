
import pandas as pd
import numpy as np
import requests
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define the API key and symbol
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

# Function to get options data for a specific date
def get_options_data(symbol, date, api_key):
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&date={date}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if 'data' in data:
        options_df = pd.DataFrame(data['data'])
        if 'strike' in options_df.columns and 'expiration' in options_df.columns:
            options_df['strike'] = pd.to_numeric(options_df['strike'], errors='coerce')
            options_df['expiration'] = pd.to_datetime(options_df['expiration'], errors='coerce')
            return options_df
        else:
            print(f"Missing necessary columns in options data for {symbol} on {date}.")
    else:
        print(f"No options data found for {symbol} on {date}.")
    
    return pd.DataFrame()

# Function to find closest options with the soonest expiration date
def find_closest_options(options_df, current_price):
    if options_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter options with the soonest expiration date
    min_expiration = options_df['expiration'].min()
    options_df = options_df[options_df['expiration'] == min_expiration]
    
    options_df['distance_to_price'] = abs(options_df['strike'] - current_price)
    
    calls_above = options_df[(options_df['type'] == 'call') & (options_df['strike'] >= current_price)].sort_values(by='strike').head(2)
    calls_below = options_df[(options_df['type'] == 'call') & (options_df['strike'] < current_price)].sort_values(by='strike', ascending=False).head(2)
    closest_calls = pd.concat([calls_above, calls_below])
    
    puts_above = options_df[(options_df['type'] == 'put') & (options_df['strike'] >= current_price)].sort_values(by='strike').head(2)
    puts_below = options_df[(options_df['type'] == 'put') & (options_df['strike'] < current_price)].sort_values(by='strike', ascending=False).head(2)
    closest_puts = pd.concat([puts_above, puts_below])
    
    closest_calls = closest_calls[['strike', 'type', 'last', 'volume', 'expiration']]
    closest_puts = closest_puts[['strike', 'type', 'last', 'volume', 'expiration']]
    
    closest_calls.columns = ['Strike Price', 'Type', 'Close Price', 'Volume', 'Expiration Date']
    closest_puts.columns = ['Strike Price', 'Type', 'Close Price', 'Volume', 'Expiration Date']
    
    return closest_calls, closest_puts

# Function to get the stock's close price for a specific date
def get_previous_close_price(symbol, date, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()

    if 'Time Series (1min)' in data:
        ts = data['Time Series (1min)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Get the trading day
        trading_day = df.index[df.index.date == date]
        if not trading_day.empty:
            close_price = df.loc[trading_day[-1], '4. close']
            return float(close_price)
    
    raise ValueError(f"No stock data found for {symbol} on {date}.")

# Determine the dates
today = datetime.now()
yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
day_before_yesterday = (today - timedelta(days=2)).strftime('%Y-%m-%d')

# Ensure we handle weekends and holidays
def get_last_trading_day(date):
    while date.weekday() > 4:  # 0 = Monday, 4 = Friday
        date -= timedelta(days=1)
    return date

try:
    # Get the last trading days
    yesterday = get_last_trading_day(today - timedelta(days=1))
    day_before_yesterday = get_last_trading_day(yesterday - timedelta(days=1))

    # Get the stock's previous close price
    previous_close_price = get_previous_close_price(symbol, yesterday.date(), api_key)
    
    # Get options data for yesterday and the previous stock market open day
    options_yesterday = get_options_data(symbol, yesterday.strftime('%Y-%m-%d'), api_key)
    options_day_before_yesterday = get_options_data(symbol, day_before_yesterday.strftime('%Y-%m-%d'), api_key)
    
    # Ensure options data is available
    if options_yesterday.empty or options_day_before_yesterday.empty:
        raise ValueError("Insufficient options data available for the given dates.")
    
    # Find the later expiration date between the two days
    min_expiration_yesterday = options_yesterday['expiration'].min()
    min_expiration_day_before_yesterday = options_day_before_yesterday['expiration'].min()
    later_expiration = max(min_expiration_yesterday, min_expiration_day_before_yesterday)

    # Filter options to only include those with the later expiration date
    options_yesterday = options_yesterday[options_yesterday['expiration'] == later_expiration]
    options_day_before_yesterday = options_day_before_yesterday[options_day_before_yesterday['expiration'] == later_expiration]

    # Find closest options for yesterday based on previous day's close price
    calls_yesterday, puts_yesterday = find_closest_options(options_yesterday, previous_close_price)
    
    # Find closest options for the day before yesterday based on the previous day's close price
    calls_day_before_yesterday, puts_day_before_yesterday = find_closest_options(options_day_before_yesterday, previous_close_price)
    
    # Combine the previous day's closing prices with yesterday's data to show them as the opening prices for yesterday
    calls_yesterday['Open Price'] = pd.to_numeric(calls_day_before_yesterday['Close Price'].values)
    puts_yesterday['Open Price'] = pd.to_numeric(puts_day_before_yesterday['Close Price'].values)

    # Convert 'Close Price' and 'Open Price' to numeric
    calls_yesterday['Close Price'] = pd.to_numeric(calls_yesterday['Close Price'])
    puts_yesterday['Close Price'] = pd.to_numeric(puts_yesterday['Close Price'])

    # Calculate the change (Close Price - Open Price)
    calls_yesterday['Change'] = calls_yesterday['Close Price'] - calls_yesterday['Open Price']
    puts_yesterday['Change'] = puts_yesterday['Close Price'] - puts_yesterday['Open Price']

    # Reorder columns to show Open Price before Close Price
    calls_yesterday = calls_yesterday[['Strike Price', 'Type', 'Open Price', 'Close Price', 'Change', 'Volume', 'Expiration Date']]
    puts_yesterday = puts_yesterday[['Strike Price', 'Type', 'Open Price', 'Close Price', 'Change', 'Volume', 'Expiration Date']]
    
    # Combine calls and puts into a single table
    combined_df = pd.concat([calls_yesterday, puts_yesterday])

    # Print combined table
    print(f"Options data for {symbol} on {yesterday.date()} (based on the previous close price ${previous_close_price:.2f}):")
    print(combined_df.to_string(index=False))

except ValueError as e:
    print(e)

# Feature Engineering for ML Model
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
lags = 20  # Adjust the number of lags as needed
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

# Define features and labels for the model
features = [
    'previous_close', '1. open', 'ma5', 'ma10', 'ma20', 'vol_change', 'high_low_diff', 'open_close_diff',
    'ema5', 'ema20', 'momentum', 'volatility', 'roc', 'macd', 'rsi'
] + [f'{col}_lag_{i}' for i in range(1, lags+1) for col in lag_columns]

# Include options data features
options_features = combined_df[['Strike Price', 'Type', 'Open Price', 'Close Price', 'Volume']]
options_features = pd.get_dummies(options_features, columns=['Type'])  # Convert 'Type' to dummy variables

# Merge stock data features and options data features
X = pd.concat([data_combined[features], options_features], axis=1)
y = combined_df['Change']

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
    'Actual_Change': y_test,
    'Predicted_Change_XGBoost': predictions_xgb
})

# Print comparison
print(comparison)

# Calculate accuracy within ±10%
accuracy = np.mean(np.abs((comparison['Actual_Change'] - comparison['Predicted_Change_XGBoost']) / comparison['Actual_Change']) <= 0.1) * 100

# Print accuracy
print(f"Prediction Accuracy: {accuracy:.2f}%")

# Plot accuracy
plt.figure(figsize=(10, 6))
categories = ['Change']
accuracies = [accuracy]

plt.bar(categories, accuracies, color=['blue'])
plt.ylim(90, 100)
plt.ylabel('Accuracy (%)')
plt.title('Prediction Accuracy for Option Price Changes')
plt.show()

# Save the model
best_xgb.save_model('best_xgb_model.json')
