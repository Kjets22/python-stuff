import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os

# Implement Focal Loss
def focal_loss(gamma=2.0, alpha=0.25):
    gamma = float(gamma)
    alpha = float(alpha)
    
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7  # Small epsilon value to avoid divide by zero errors
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return loss
    
    return focal_loss_fixed

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

# Adjust labeling logic to identify confirmed breakouts and their types
def label_breakouts(df, min_price_change=0.005, time_window=60):
    # Initialize the 'breakout_type' column with a default value
    df['breakout_type'] = 0  # 0 for 'No Breakout'

    # Ensure necessary columns are present
    required_columns = ['close_1min', 'momentum_1min', 'rsi_1min', 'macd_1min', 'bb_high_1min', 
                        'momentum_5min', 'momentum_15min', 'vol_change_1min']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Calculate percentage price change
    price_change = (df['close_1min'].shift(-time_window) - df['close_1min']) / df['close_1min']

    # Breakout conditions
    upward_breakout = (
        (price_change >= min_price_change) & \
        (df['momentum_1min'] > 0) & \
        (df['rsi_1min'] > 60) & \
        (df['macd_1min'] > -1) & \
        (df['close_1min'] > df['bb_high_1min'])
    )

    reversal_breakout = (
        (price_change >= min_price_change) & \
        (df['momentum_1min'] > 0) & \
        (df['momentum_5min'] < 0) & \
        (df['macd_1min'].diff() > -0.5)
    )

    steady_climb = (
        (price_change >= min_price_change) & \
        (df['momentum_1min'] > 0) & \
        (df['momentum_5min'] > 0) & \
        (df['momentum_15min'] > 0)
    )

    resistance_level = df['close_1min'].rolling(window=20).max()
    resistance_breakout = (
        (df['close_1min'] > resistance_level * 0.995) & \
        (price_change >= min_price_change) & \
        (df['vol_change_1min'] > 1.2 * df['vol_change_1min'].rolling(window=20).mean())
    )

    # Assign labels
    df.loc[upward_breakout, 'breakout_type'] = 1
    df.loc[reversal_breakout, 'breakout_type'] = 2
    df.loc[steady_climb, 'breakout_type'] = 3
    df.loc[resistance_breakout, 'breakout_type'] = 4

    return df

# Prepare data for LSTM
def prepare_lstm_data(df, features, target, time_steps=9000):
    X, y = [], []
    data_length = len(df)
    for i in range(time_steps, data_length):
        X.append(df[features].iloc[i-time_steps:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# Build the LSTM model with enhanced architecture
def build_lstm_model(input_shape, num_classes, units_1=256, units_2=128, units_3=64, dropout_rate=0.4, learning_rate=0.0005):
    model = Sequential()
    model.add(LSTM(units=units_1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_3))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model with focal loss
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
    return model

# File path to your combined data
data_file = 'combined_data.txt'

# Define the chunk size and overlap
chunk_size = 10000  # Adjust based on your memory capacity
overlap_size = 9000  # Overlap to prevent data leakage between chunks

# Initialize variables
time_steps = 10  # Number of time steps for LSTM
batch_size = 32  # Adjust batch size as needed
features = None
num_classes = 5  # Number of classes
scaler = StandardScaler()
model_initialized = False

# Initialize variables to collect evaluation metrics
all_y_true = []
all_y_pred = []

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
    chunk = label_breakouts(chunk)
    
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
    
    # Normalize the data
    chunk[features] = scaler.fit_transform(chunk[features])
    
    # Prepare data for LSTM
    X_chunk, y_chunk = prepare_lstm_data(chunk, features, 'breakout_type', time_steps=time_steps)
    
    # Ensure y_chunk is integer
    y_chunk = y_chunk.astype(int)
    
    # Skip if there's no data
    if len(X_chunk) == 0:
        print(f"Chunk {chunk_number} skipped due to insufficient data after preparing LSTM sequences.")
        continue
    
    # One-hot encode the labels
    y_chunk_categorical = to_categorical(y_chunk, num_classes=num_classes)
    
    # Handle class imbalance using class weights
    y_integers = np.argmax(y_chunk_categorical, axis=1)
    # Compute class weights for classes present in y_integers
    classes_in_y = np.unique(y_integers)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=classes_in_y,
        y=y_integers
    )
    class_weights_present = dict(zip(classes_in_y, class_weights_array))
    # Ensure class_weights includes all classes
    all_classes = np.array([0, 1, 2, 3, 4])
    max_weight = max(class_weights_array)
    class_weights = {}
    for cls in all_classes:
        if cls in class_weights_present:
            class_weights[cls] = class_weights_present[cls]
        else:
            class_weights[cls] = max_weight  # Assign maximum weight to missing classes
    
    # Build or load the model
    if not model_initialized:
        lstm_model = build_lstm_model(input_shape=(X_chunk.shape[1], X_chunk.shape[2]), num_classes=num_classes)
        model_initialized = True
    else:
        # Load the best model weights if available
        if os.path.exists('best_model.h5'):
            lstm_model.load_weights('best_model.h5')
    
    # Train the model on the current chunk
    lstm_model.fit(
        X_chunk, y_chunk_categorical,
        epochs=3,  # Use a small number of epochs per chunk
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the model weights
    lstm_model.save_weights('best_model.h5')
    
    # Evaluate on the current chunk (optional)
    y_pred_prob = lstm_model.predict(X_chunk)
    y_pred = np.argmax(y_pred_prob, axis=1)
    all_y_true.extend(y_chunk)
    all_y_pred.extend(y_pred)
    
    # Clear variables to free memory
    del chunk, X_chunk, y_chunk, y_chunk_categorical
    import gc
    gc.collect()

# Evaluate the model on the collected data
if len(all_y_true) > 0:
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    # Generate classification report
    print(classification_report(
        y_true, y_pred, 
        labels=[0, 1, 2, 3, 4], 
        target_names=[
            'No Breakout', 'Upward Breakout', 'Reversal Breakout', 'Steady Climb', 'Resistance Breakout'
        ],
        zero_division=0
    ))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot the confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Breakout', 'Upward', 'Reversal', 'Steady Climb', 'Resistance'],
                yticklabels=['No Breakout', 'Upward', 'Reversal', 'Steady Climb', 'Resistance'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
else:
    print("No data collected for evaluation.")
