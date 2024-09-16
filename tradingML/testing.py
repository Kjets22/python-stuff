
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import random

# Delete existing .h5 files to prevent loading previous weights
if os.path.exists('best_lstm_model.h5'):
    os.remove('best_lstm_model.h5')

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
            feature_dict[f'vol_change_{interval}'] = df[f'volume_{interval}'].pct_change(fill_method=None).fillna(0)
            feature_dict[f'high_low_diff_{interval}'] = df[f'high_{interval}'] - df[f'low_{interval}']
            feature_dict[f'open_close_diff_{interval}'] = df[f'open_{interval}'] - df[close_col]
            feature_dict[f'ema5_{interval}'] = df[close_col].ewm(span=5, adjust=False).mean()
            feature_dict[f'ema20_{interval}'] = df[close_col].ewm(span=20, adjust=False).mean()
            feature_dict[f'momentum_{interval}'] = df[close_col] - df[close_col].shift(4).fillna(0)
            feature_dict[f'volatility_{interval}'] = df[close_col].rolling(window=5, min_periods=1).std()
            feature_dict[f'roc_{interval}'] = df[close_col].pct_change(periods=10, fill_method=None)
            feature_dict[f'ema12_{interval}'] = df[close_col].ewm(span=12, adjust=False).mean()
            feature_dict[f'ema26_{interval}'] = df[close_col].ewm(span=26, adjust=False).mean()
            feature_dict[f'macd_{interval}'] = feature_dict[f'ema12_{interval}'] - feature_dict[f'ema26_{interval}']
            
            # New indicators
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.volatility import BollingerBands

            # Handle cases where there are not enough data points
            try:
                feature_dict[f'rsi_{interval}'] = RSIIndicator(df[close_col], window=14).rsi()
            except Exception:
                feature_dict[f'rsi_{interval}'] = np.nan
            try:
                bb_indicator = BollingerBands(df[close_col], window=20, window_dev=2)
                feature_dict[f'bb_high_{interval}'] = bb_indicator.bollinger_hband()
                feature_dict[f'bb_low_{interval}'] = bb_indicator.bollinger_lband()
            except Exception:
                feature_dict[f'bb_high_{interval}'] = np.nan
                feature_dict[f'bb_low_{interval}'] = np.nan
            try:
                stoch_indicator = StochasticOscillator(df[f'high_{interval}'], df[f'low_{interval}'], df[close_col], window=14, smooth_window=3)
                feature_dict[f'stoch_k_{interval}'] = stoch_indicator.stoch()
                feature_dict[f'stoch_d_{interval}'] = stoch_indicator.stoch_signal()
            except Exception:
                feature_dict[f'stoch_k_{interval}'] = np.nan
                feature_dict[f'stoch_d_{interval}'] = np.nan
                
            # Interaction terms
            feature_dict[f'ma5_ma20_ratio_{interval}'] = feature_dict[f'ma5_{interval}'] / feature_dict[f'ma20_{interval}']
            feature_dict[f'ema5_ema20_ratio_{interval}'] = feature_dict[f'ema5_{interval}'] / feature_dict[f'ema20_{interval}']
            feature_dict[f'vol_change_momentum_{interval}'] = feature_dict[f'vol_change_{interval}'] * feature_dict[f'momentum_{interval}']

    # Concatenate all the new features to the original dataframe
    feature_df = pd.DataFrame(feature_dict, index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    return df

# Adjust labeling logic to identify breakouts
def label_breakouts(df, min_price_change=0.005, max_price_drop=0.001, time_window=120):
    # Initialize the 'breakout_type' column with a default value
    df['breakout_type'] = 0  # 0 for 'No Breakout'

    # Ensure necessary columns are present
    required_columns = ['close_1min']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Calculate future price changes over the time window
    future_prices = df['close_1min'].shift(-time_window)
    price_change = (future_prices - df['close_1min']) / df['close_1min']
    max_price_increase = df['close_1min'].rolling(window=time_window).max().shift(-time_window)
    max_price_decrease = df['close_1min'].rolling(window=time_window).min().shift(-time_window)
    drawdown = (max_price_decrease - df['close_1min']) / df['close_1min']
    drawup = (max_price_increase - df['close_1min']) / df['close_1min']

    # Upward Breakout
    upward_breakout = (
        (price_change >= min_price_change) & \
        (drawdown >= -max_price_drop)
    )

    # Downward Breakout
    downward_breakout = (
        (price_change <= -min_price_change) & \
        (drawup <= max_price_drop)
    )

    # Assign labels
    df.loc[upward_breakout, 'breakout_type'] = 1
    df.loc[downward_breakout, 'breakout_type'] = 2

    return df

# Prepare data for LSTM (used for both LSTM and XGBoost)
def prepare_lstm_data(df, features, target, time_steps=1000):
    X, y = [], []
    data_length = len(df)
    for i in range(time_steps, data_length):
        X.append(df[features].iloc[i-time_steps:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# Build the LSTM model
def build_lstm_model(input_shape, num_classes, units_1=128, units_2=64, units_3=32, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=units_1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_3))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# File path to your combined data
data_file = 'combined_data.txt'

# Define the chunk size and overlap
chunk_size = 5000  # Increase chunk size to accommodate larger time_steps
overlap_size = 1000  # Set overlap size to match time_steps

# Initialize variables
time_steps = 1000  # Set time_steps to 1000
batch_size = 32  # Adjust batch size as needed
features = None
num_classes = 3  # Classes: 0 (No Breakout), 1 (Upward Breakout), 2 (Downward Breakout)
scaler = StandardScaler()
model_initialized = False

# Initialize variables to collect evaluation metrics
all_y_true = []
all_y_pred_lstm = []
all_y_pred_xgb = []

# Read and process data in chunks
chunk_iterator = pd.read_csv(data_file, sep='\t', chunksize=chunk_size - overlap_size, parse_dates=['timestamp'])
chunk_number = 0
prev_chunk_tail = None

for chunk in chunk_iterator:
    chunk_number += 1
    print(f"Processing chunk {chunk_number}")
    
    # If there is a previous chunk's tail, concatenate it
    if prev_chunk_tail is not None:
        chunk = pd.concat([prev_chunk_tail, chunk], ignore_index=True)
    
    # Keep the last 'overlap_size' data points for the next chunk
    prev_chunk_tail = chunk.iloc[-overlap_size:].copy()
    
    # Set the index
    chunk = chunk.set_index('timestamp')
    
    # Perform feature engineering
    chunk = add_enhanced_features(chunk)
    
    # Label breakouts
    chunk = label_breakouts(chunk, min_price_change=0.005, max_price_drop=0.001, time_window=120)
    
    # Handle infinity or extremely large values
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.fillna(0, inplace=True)
    
    # Define features after first chunk
    if features is None:
        features = [col for col in chunk.columns if col != 'breakout_type']
    
    # Check if we have enough data points after time_steps
    if len(chunk) <= time_steps:
        print(f"Chunk {chunk_number} skipped due to insufficient data after time_steps.")
        continue
    
    # Prepare the dataset
    chunk_data = chunk.copy()
    
    # Balance the dataset
    breakout_indices = chunk_data[chunk_data['breakout_type'] > 0].index
    no_breakout_indices = chunk_data[chunk_data['breakout_type'] == 0].index
    
    num_breakouts = len(breakout_indices)
    num_no_breakouts = min(len(no_breakout_indices), num_breakouts * 10)
    
    # Skip the chunk if there are no breakouts
    if num_breakouts == 0:
        print(f"Chunk {chunk_number} skipped due to no breakout instances.")
        continue
    
    # Randomly select 'No Breakout' instances
    selected_no_breakout_indices = np.random.choice(no_breakout_indices, size=num_no_breakouts, replace=False)
    
    # Combine indices
    combined_indices = np.concatenate((breakout_indices, selected_no_breakout_indices))
    
    # Create balanced chunk
    balanced_chunk = chunk_data.loc[combined_indices]
    balanced_chunk = balanced_chunk.sort_index()  # Ensure data is in chronological order
    
    # Normalize the data
    balanced_chunk[features] = scaler.fit_transform(balanced_chunk[features])
    
    # Prepare data for both LSTM and XGBoost using the same function
    X_chunk, y_chunk = prepare_lstm_data(balanced_chunk, features, 'breakout_type', time_steps=time_steps)
    
    # Ensure y_chunk is integer
    y_chunk = y_chunk.astype(int)
    
    # Skip if there's no data
    if len(X_chunk) == 0:
        print(f"Chunk {chunk_number} skipped due to insufficient data after preparing sequences.")
        continue
    
    # Reshape X_chunk for XGBoost
    n_samples, time_steps, n_features = X_chunk.shape
    X_chunk_xgb = X_chunk.reshape((n_samples, time_steps * n_features))
    y_chunk_xgb = y_chunk  # Labels remain the same
    
    # One-hot encode the labels for LSTM
    y_chunk_categorical = to_categorical(y_chunk, num_classes=num_classes)
    
    # Split data into training and testing sets
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
        X_chunk, y_chunk_categorical, test_size=0.2, random_state=42, stratify=y_chunk
    )
    
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_chunk_xgb, y_chunk_xgb, test_size=0.2, random_state=42, stratify=y_chunk_xgb
    )
    
    # Build the models if not already initialized
    if not model_initialized:
        lstm_model = build_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), num_classes=num_classes)
        # Initialize XGBoost model without 'use_label_encoder'
        xgb_model = XGBClassifier(eval_metric='mlogloss', num_class=num_classes)
        model_initialized = True
    
    # Train the LSTM model on the current chunk
    lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=3,  # Use a small number of epochs per chunk
        batch_size=batch_size,
        verbose=1
    )
    
    # Save the LSTM model weights
    lstm_model.save_weights('best_lstm_model.h5')
    
    # Train the XGBoost model on the current chunk
    xgb_model.fit(X_train_xgb, y_train_xgb)
    
    # Evaluate LSTM model on the test set
    y_pred_lstm_prob = lstm_model.predict(X_test_lstm)
    y_pred_lstm = np.argmax(y_pred_lstm_prob, axis=1)
    y_test_lstm_labels = np.argmax(y_test_lstm, axis=1)
    all_y_true.extend(y_test_lstm_labels)
    all_y_pred_lstm.extend(y_pred_lstm)
    
    # Evaluate XGBoost model on the test set
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    all_y_pred_xgb.extend(y_pred_xgb)
    
    # Clear variables to free memory
    del chunk, X_chunk, y_chunk, y_chunk_categorical, X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm
    del X_chunk_xgb, y_chunk_xgb, X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb
    import gc
    gc.collect()

# Evaluate the models on the collected data
if len(all_y_true) > 0:
    y_true = np.array(all_y_true)
    y_pred_lstm = np.array(all_y_pred_lstm)
    y_pred_xgb = np.array(all_y_pred_xgb)
    
    # Define class names
    class_names = ['No Breakout', 'Upward Breakout', 'Downward Breakout']
    
    # LSTM Model Evaluation
    print("LSTM Model Evaluation:")
    print(classification_report(
        y_true, y_pred_lstm, 
        labels=[0, 1, 2], 
        target_names=class_names,
        zero_division=0
    ))
    print("Confusion Matrix:")
    conf_matrix_lstm = confusion_matrix(y_true, y_pred_lstm, labels=[0, 1, 2])
    print(conf_matrix_lstm)
    
    # XGBoost Model Evaluation
    print("\nXGBoost Model Evaluation:")
    print(classification_report(
        y_true, y_pred_xgb, 
        labels=[0, 1, 2], 
        target_names=class_names,
        zero_division=0
    ))
    print("Confusion Matrix:")
    conf_matrix_xgb = confusion_matrix(y_true, y_pred_xgb, labels=[0, 1, 2])
    print(conf_matrix_xgb)
    
    # Plot the confusion matrices
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # LSTM Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_lstm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('LSTM Confusion Matrix')
    plt.show()
    
    # XGBoost Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('XGBoost Confusion Matrix')
    plt.show()
else:
    print("No data collected for evaluation.")
