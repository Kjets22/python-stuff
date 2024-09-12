import os
import psutil
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Print when the process starts
print("Process started...")

# Ensure TensorFlow uses the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from using all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth set successfully.")
    except RuntimeError as e:
        print(e)

# Function to load data in chunks
def load_data_in_chunks(file_path, chunk_size=50000):
    chunks = []
    chunk_num = 0
    for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size, index_col='timestamp', parse_dates=True):
        print(f"Loading chunk {chunk_num}, size: {chunk.shape}")
        chunk_num += 1
        chunks.append(chunk)
    return pd.concat(chunks, axis=0)

# Print memory usage for tracking progress
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Feature engineering
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
    feature_df = pd.concat(feature_dict.values(), axis=1)
    feature_df.columns = feature_dict.keys()
    return pd.concat([df, feature_df], axis=1)

# Breakout labeling logic
def label_breakouts(df, min_price_change=2.0, time_window=100):  
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
    
    resistance_level = df['close_1min'].rolling(window=20).max()  
    breakout_dict['resistance_breakout'] = (
        (df['close_1min'] > resistance_level) & \
        (df['close_1min'].shift(-time_window) - df['close_1min']) >= min_price_change) & \
        (df['vol_change_1min'] > 1.5 * df['vol_change_1min'].rolling(window=20).mean())

    breakout_df = pd.DataFrame(breakout_dict)
    breakout_df['breakout_type'] = breakout_df.any(axis=1).astype(int)
    return pd.concat([df, breakout_df], axis=1).drop(
        ['upward_breakout', 'reversal_breakout', 'steady_climb', 'resistance_breakout'], axis=1)

# Preparing LSTM data
def prepare_lstm_data(df, features, target, time_steps=100):
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df[features].iloc[i - time_steps:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# Model Building
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

# Main execution
if __name__ == "__main__":
    print("Starting data processing...")
    
    # Load data in chunks
    file_path = 'combined_data.txt'
    chunk_size = 50000
    features = None

    model = build_lstm_model(input_shape=(100, 40))  # Adjust input shape accordingly

    for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size, index_col='timestamp', parse_dates=True):
        print(f"Processing chunk {chunk.shape}")
        print_memory_usage()

        # Feature engineering
        chunk = add_enhanced_features(chunk)

        # Label breakouts
        chunk = label_breakouts(chunk)
        
        # Count breakouts in this chunk
        num_breakouts = chunk['breakout_type'].sum()
        print(f"Number of breakouts in this chunk: {num_breakouts}")

        # Prepare LSTM data
        if features is None:
            features = [col for col in chunk.columns if col != 'breakout_type']

        X_train, y_train = prepare_lstm_data(chunk, features, 'breakout_type')
        X_train_scaled = StandardScaler().fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

        # Train the model on the chunk using the GPU
        with tf.device('/GPU:0'):
            print("Training on batch...")
            model.train_on_batch(X_train_scaled, y_train)

    print("Training complete.")
    print_memory_usage()
