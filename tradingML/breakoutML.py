import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import MACD
import tensorflow as tf

# Polygon.io API key
api_key = 'LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg'

# Symbol to fetch data for
symbol = 'SPY'

# Function to fetch data for multiple timeframes from Polygon.io
def get_intraday_data_polygon(symbol, interval, api_key, start_date, end_date, limit=50000):
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    interval_map = {
        '1min': '1',
        '5min': '5',
        '15min': '15',
        '30min': '30',
        '60min': '60'
    }
    
    all_data = []
    current_date = end_date

    while True:
        url = f"{base_url}/{symbol}/range/{interval_map[interval]}/minute/{start_date}/{current_date}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"API Request failed with status code {response.status_code}: {response.text}")
        
        data = response.json().get('results', [])
        if not data:
            break
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        all_data.append(df)
        
        # Move to the next batch by changing the current_date to the earliest timestamp in this batch
        current_date = df.index.min().strftime('%Y-%m-%d')
        
        if df.index.min() <= pd.to_datetime(start_date):
            break
    
    if all_data:
        df = pd.concat(all_data)
        df.rename(columns={
            'o': f'open_{interval}',
            'h': f'high_{interval}',
            'l': f'low_{interval}',
            'c': f'close_{interval}',
            'v': f'volume_{interval}'
        }, inplace=True)
        
        return df[[f'open_{interval}', f'high_{interval}', f'low_{interval}', f'close_{interval}', f'volume_{interval}']]
    else:
        return pd.DataFrame()

# Define the intervals to fetch data for
intervals = ['1min', '5min', '15min', '30min', '60min']
start_date = '2018-01-01'  # Extend the date range further back
end_date = '2023-01-01'

# Fetch data for SPY and combine into one DataFrame
dataframes = [get_intraday_data_polygon(symbol, interval, api_key, start_date, end_date) for interval in intervals]
data_combined = pd.concat(dataframes, axis=1)
data_combined.dropna(inplace=True)

# Feature engineering with additional indicators
def add_enhanced_features(df):
    feature_dict = {}
    
    for interval in ['1min', '5min', '15min', '30min', '60min']:
        close_col = f'close_{interval}'
        if close_col in df.columns:
            feature_dict[f'previous_close_{interval}'] = df[close_col].shift(1)
            feature_dict[f'price_change_{interval}'] = df[close_col] - df[f'open_{interval}']
            feature_dict[f'ma5_{interval}'] = df[close_col].rolling(window=5, min_periods=1).mean()
            feature_dict[f'ma10_{interval}'] = df[close_col].rolling(window=10, min_periods=1).mean()
            feature_dict[f'ma20_{interval}'] = df[close_col].rolling(window=20, min_periods=1).mean()
            feature_dict[f'vol_change_{interval}'] = df[f'volume_{interval}'].pct_change().fillna(0)
            feature_dict[f'high_low_diff_{interval}'] = df[f'high_{interval}'] - df[f'low_{interval}']
            feature_dict[f'open_close_diff_{interval}'] = df[f'open_{interval}'] - df[close_col]
            feature_dict[f'ema5_{interval}'] = df[close_col].ewm(span=5, adjust=False).mean()
            feature_dict[f'ema20_{interval}'] = df[close_col].ewm(span=20, adjust=False).mean()
            feature_dict[f'momentum_{interval}'] = df[close_col] - df[close_col].shift(4).fillna(0)
            feature_dict[f'volatility_{interval}'] = df[close_col].rolling(window=5, min_periods=1).std()
            feature_dict[f'roc_{interval}'] = df[close_col].pct_change(periods=10)
            feature_dict[f'ema12_{interval}'] = df[close_col].ewm(span=12, adjust=False).mean()
            feature_dict[f'ema26_{interval}'] = df[close_col].ewm(span=26, adjust=False).mean()
            feature_dict[f'macd_{interval}'] = feature_dict[f'ema12_{interval}'] - feature_dict[f'ema26_{interval}']
            
            # New indicators
            feature_dict[f'rsi_{interval}'] = RSIIndicator(df[close_col], window=14).rsi()
            bb_indicator = BollingerBands(df[close_col], window=20, window_dev=2)
            feature_dict[f'bb_high_{interval}'] = bb_indicator.bollinger_hband()
            feature_dict[f'bb_low_{interval}'] = bb_indicator.bollinger_lband()
            stoch_indicator = StochasticOscillator(df[f'high_{interval}'], df[f'low_{interval}'], df[close_col], window=14, smooth_window=3)
            feature_dict[f'stoch_k_{interval}'] = stoch_indicator.stoch()
            feature_dict[f'stoch_d_{interval}'] = stoch_indicator.stoch_signal()
            
            # Interaction terms
            feature_dict[f'ma5_ma20_ratio_{interval}'] = feature_dict[f'ma5_{interval}'] / feature_dict[f'ma20_{interval}']
            feature_dict[f'ema5_ema20_ratio_{interval}'] = feature_dict[f'ema5_{interval}'] / feature_dict[f'ema20_{interval}']
            feature_dict[f'vol_change_momentum_{interval}'] = feature_dict[f'vol_change_{interval}'] * feature_dict[f'momentum_{interval}']

    # Concatenate all the new features to the original dataframe
    feature_df = pd.concat(feature_dict.values(), axis=1)
    feature_df.columns = feature_dict.keys()
    
    return pd.concat([df, feature_df], axis=1)

data_combined = add_enhanced_features(data_combined)

# Adjust labeling logic to identify confirmed breakouts with at least $2 change within an hour
def label_breakouts(df, min_price_change=2.0, time_window=100):  # Set min_price_change to $2, time_window to 60 minutes
    breakout_dict = {}

    breakout_dict['upward_breakout'] = (
        (df['close_1min'].shift(-time_window) - df['close_1min']) >= min_price_change) & \
        (df['momentum_1min'] > 0) & \
        (df['rsi_1min'] > 70) & \
        (df['macd_1min'] > 0) & \
        (df['close_1min'] > df['bb_high_1min'])
    
    breakout_dict['reversal_breakout'] = (
        (df['close_1min'].shift(-time_window) - df['close_1min']) >= min_price_change) & \
        (df['momentum_1min'] > 0) & \
        (df['momentum_5min'] < 0) & \
        (df['macd_1min'].diff() > 0)
    
    breakout_dict['steady_climb'] = (
        (df['close_1min'].shift(-time_window) - df['close_1min']) >= min_price_change) & \
        (df['momentum_1min'] > 0) & \
        (df['momentum_5min'] > 0) & \
        (df['momentum_15min'] > 0)
    
    resistance_level = df['close_1min'].rolling(window=20).max()  # Example resistance level
    breakout_dict['resistance_breakout'] = (
        (df['close_1min'] > resistance_level) & \
        (df['close_1min'].shift(-time_window) - df['close_1min']) >= min_price_change) & \
        (df['vol_change_1min'] > 1.5 * df['vol_change_1min'].rolling(window=20).mean())

    breakout_df = pd.DataFrame(breakout_dict)

    # Combine all breakout conditions into one label
    breakout_df['breakout_type'] = breakout_df.any(axis=1).astype(int)

    return pd.concat([df, breakout_df], axis=1).drop(
        ['upward_breakout', 'reversal_breakout', 'steady_climb', 'resistance_breakout'], axis=1)

data_combined = label_breakouts(data_combined)

# Function to extract breakout times
def get_breakout_times(df):
    breakout_times = df[df['breakout_type'] == 1].index.tolist()
    return breakout_times

breakout_times = get_breakout_times(data_combined)

# Print the times of each breakout
print("Breakout Times:")
for time in breakout_times:
    print(time)

# Print the number of confirmed breakouts in the dataset
num_breakouts = data_combined['breakout_type'].sum()
print(f"Number of confirmed breakouts in the dataset: {num_breakouts}")

# Handle infinity or extremely large values
data_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
data_combined.fillna(0, inplace=True)

# Prepare LSTM input data
def prepare_lstm_data(df, features, target, time_steps=100):
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df[features].iloc[i-time_steps:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# Extract features and target
features = [col for col in data_combined.columns if col != 'breakout_type']
X_train, y_train = prepare_lstm_data(data_combined, features, 'breakout_type')

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

# Split the data using stratified sampling to ensure balanced classes in both training and testing datasets
X_train_prepared, X_test_prepared, y_train_prepared, y_test_prepared = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=42, stratify=y_train)

# Build the LSTM model
def build_lstm_model(input_shape, units_1=256, units_2=128, units_3=64, dropout_rate=0.3, optimizer=Adam, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=units_1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_3, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizer(learning_rate=learning_rate, decay=1e-6)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Use GridSearchCV for hyperparameter tuning
param_grid = {
    'units_1': [128, 256],
    'units_2': [64, 128],
    'units_3': [32, 64],
    'dropout_rate': [0.3, 0.4, 0.5],
    'optimizer': [Adam],
    'learning_rate': [0.001, 0.0005, 0.0001]
}

model = KerasClassifier(build_fn=build_lstm_model, input_shape=(X_train_prepared.shape[1], X_train_prepared.shape[2]), epochs=50, batch_size=32, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid.fit(X_train_prepared, y_train_prepared)

# Print best parameters and score
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")

# Train final model with best parameters
best_model = grid_result.best_estimator_
best_model.fit(X_train_prepared, y_train_prepared)

# Final evaluation on test data
y_pred_prob = best_model.predict(X_test_prepared)
y_pred = (y_pred_prob > 0.5).astype(int)

print(f"Final Accuracy: {accuracy_score(y_test_prepared, y_pred)}")
print(classification_report(y_test_prepared, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test_prepared, y_pred, labels=[0, 1]))

if len(np.unique(y_test_prepared)) > 1:
    roc_auc = roc_auc_score(y_test_prepared, y_pred_prob)
    print(f"ROC AUC Score: {roc_auc}")
else:
    print("ROC AUC Score is undefined due to only one class being present in y_test.")

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test_prepared, y_pred, labels=[0, 1])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Breakout', 'Breakout'], yticklabels=['No Breakout', 'Breakout'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
