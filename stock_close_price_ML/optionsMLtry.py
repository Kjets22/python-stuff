import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

# Fetch data from Alpha Vantage
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

ts = TimeSeries(key=api_key, output_format='pandas')

# Define date range for 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

# Fetching historical daily stock data
stock_data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')

# Filter stock data to match the date range
stock_data = stock_data[(stock_data.index >= str(start_date)) & (stock_data.index <= str(end_date))]
stock_data = stock_data.reset_index().rename(columns={'date': 'datetime'})

# Feature Engineering on stock data
stock_data['previous_close'] = stock_data['4. close'].shift(1)
stock_data['price_change'] = stock_data['4. close'] - stock_data['1. open']
stock_data['ma5'] = stock_data['4. close'].rolling(window=5, min_periods=1).mean()
stock_data['ma10'] = stock_data['4. close'].rolling(window=10, min_periods=1).mean()
stock_data['ma20'] = stock_data['4. close'].rolling(window=20, min_periods=1).mean()
stock_data['vol_change'] = stock_data['6. volume'].pct_change().fillna(0)
stock_data['high_low_diff'] = stock_data['2. high'] - stock_data['3. low']
stock_data['open_close_diff'] = stock_data['1. open'] - stock_data['4. close']

# Add more technical indicators
stock_data['ema5'] = stock_data['4. close'].ewm(span=5, adjust=False).mean()
stock_data['ema20'] = stock_data['4. close'].ewm(span=20, adjust=False).mean()
stock_data['momentum'] = stock_data['4. close'] - stock_data['4. close'].shift(4).fillna(0)
stock_data['volatility'] = stock_data['4. close'].rolling(window=5, min_periods=1).std()

# Add additional features
stock_data['roc'] = stock_data['4. close'].pct_change(periods=10)  # Rate of change
stock_data['ema12'] = stock_data['4. close'].ewm(span=12, adjust=False).mean()
stock_data['ema26'] = stock_data['4. close'].ewm(span=26, adjust=False).mean()
stock_data['macd'] = stock_data['ema12'] - stock_data['ema26']

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

stock_data['rsi'] = calculate_rsi(stock_data['4. close'])

# Add lag features
lags = 10  # Adjust the number of lags as needed
lag_columns = ['4. close', '2. high', '3. low', '6. volume']
lagged_data = pd.concat([stock_data[lag_columns].shift(i).add_suffix(f'_lag_{i}') for i in range(1, lags+1)], axis=1)
stock_data = pd.concat([stock_data, lagged_data], axis=1)

# Drop rows with any missing values after adding lag features
stock_data = stock_data.dropna()

# Check the number of samples we have after feature engineering
print(f'Number of samples after feature engineering: {len(stock_data)}')

if len(stock_data) <= 1:
    raise ValueError("Not enough data samples after feature engineering. Please check the data preprocessing steps and ensure there is sufficient data.")

# Replace infinite values and very large values
stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
stock_data.fillna(0, inplace=True)

### Fetching and Integrating Historical Option Data for multiple dates
historical_option_api_url = "https://www.alphavantage.co/query"
historical_option_params = {
    "function": "HISTORICAL_OPTIONS",
    "symbol": symbol,
    "apikey": api_key
}

# Collect option data for multiple dates
all_options_data = []

for single_date in pd.date_range(start=start_date, end=end_date, freq='ME'):
    historical_option_params["date"] = single_date.strftime('%Y-%m-%d')
    response = requests.get(historical_option_api_url, params=historical_option_params)
    option_data = response.json()

    if 'data' in option_data:
        all_options_data.extend(option_data['data'])

# Convert to DataFrame
options_df = pd.DataFrame(all_options_data)

if not options_df.empty:
    # Filter for call and put options
    calls_df = options_df[options_df['type'] == 'call'].copy()
    puts_df = options_df[options_df['type'] == 'put'].copy()

    # Assuming we want to use the last price and the datetime
    calls_df.loc[:, 'datetime'] = pd.to_datetime(calls_df['date'])
    calls_df = calls_df[['datetime', 'strike', 'last']].rename(columns={'last': 'call_price'})

    puts_df.loc[:, 'datetime'] = pd.to_datetime(puts_df['date'])
    puts_df = puts_df[['datetime', 'strike', 'last']].rename(columns={'last': 'put_price'})

    # Print date ranges for debugging
    print(f"Stock data date range: {stock_data['datetime'].min()} to {stock_data['datetime'].max()}")
    print(f"Option calls data date range: {calls_df['datetime'].min()} to {calls_df['datetime'].max()}")
    print(f"Option puts data date range: {puts_df['datetime'].min()} to {puts_df['datetime'].max()}")

    # Merge option data with the stock data
    data_combined = pd.merge(stock_data, calls_df, on='datetime', how='inner')
    data_combined = pd.merge(data_combined, puts_df, on='datetime', how='inner')

    # Ensure that the merged data is not empty
    if data_combined.empty:
        raise ValueError("Merged data is empty. Check date ranges and ensure data is available.")

    # Feature Engineering for Option Data
    data_combined['call_price_change'] = data_combined['call_price'].pct_change().fillna(0)
    data_combined['put_price_change'] = data_combined['put_price'].pct_change().fillna(0)

    # Define features and labels
    features = [
        'previous_close', '1. open', 'ma5', 'ma10', 'ma20', 'vol_change', 'high_low_diff', 'open_close_diff',
        'ema5', 'ema20', 'momentum', 'volatility', 'roc', 'macd', 'rsi',
        'call_price_change', 'put_price_change'
    ] + [f'{col}_lag_{i}' for i in range(1, lags+1) for col in lag_columns]

    # Our target is to predict which option will increase the most
    data_combined['option_max_increase'] = data_combined[['call_price_change', 'put_price_change']].idxmax(axis=1)

    X = data_combined[features]
    y = data_combined['option_max_increase']

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
        'Actual': y_test,
        'Predicted': predictions_xgb
    })

    # Select 10 random test examples to print
    sampled_comparison = comparison.sample(n=10, random_state=42)

    # Print selected test examples
    for index, row in sampled_comparison.iterrows():
        print(f"Example {index + 1}:")
        print(f"  Actual: {row['Actual']}")
        print(f"  Predicted: {row['Predicted']}")
        print()

    # Calculate accuracy within ±10%
    accuracy = np.mean(y_test == predictions_xgb) * 100

    # Print accuracy
    print(f"Prediction Accuracy: {accuracy:.2f}%")

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    categories = ['Call Option', 'Put Option']
    accuracies = [accuracy]  # Simplified for illustration

    plt.bar(categories, accuracies, color=['blue', 'orange'])
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Prediction Accuracy for Options')
    plt.show()

    # Save the model
    best_xgb.save_model('best_xgb_model.json')

else:
    print("Error: The structure of the option data does not contain 'data'")
