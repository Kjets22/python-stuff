import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from collections import Counter
import tensorflow as tf
import os
import random
import gc  # Garbage Collector for memory management
import logging  # For detailed logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Suppress TensorFlow INFO and WARNING logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Options to suppress logs from TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Remove existing .h5 model weights file to prevent loading old weights
if os.path.exists('best_lstm_model.h5'):
    os.remove('best_lstm_model.h5')
    logging.info("Existing LSTM model weights deleted.")

# Feature engineering function that adds technical indicators to the dataframe
def add_enhanced_features(df):
    """
    Adds technical indicators like moving averages, price change, 
    volatility, and more to the dataframe for various intervals (1min, 5min, etc.)

    Parameters:
    - df (pd.DataFrame): The input dataframe containing stock data.

    Returns:
    - pd.DataFrame: The dataframe with new engineered features.
    """
    feature_dict = {}
    
    # Loop through different time intervals (1min, 5min, etc.) and add features
    for interval in ['1min', '5min', '15min', '30min', '60min']:
        close_col = f'close_{interval}'  # Get the close price column for each interval
        if close_col in df.columns:
            # Calculate various features like price changes, moving averages, EMA, volatility, etc.
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
            
            # Add RSI, Bollinger Bands, and Stochastic Oscillator
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.volatility import BollingerBands

            # Add technical indicators like RSI and Bollinger Bands
            try:
                rsi = RSIIndicator(df[close_col], window=14)
                feature_dict[f'rsi_{interval}'] = rsi.rsi()
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
                
            # Calculate interaction terms between moving averages and other indicators
            feature_dict[f'ma5_ma20_ratio_{interval}'] = feature_dict[f'ma5_{interval}'] / feature_dict[f'ma20_{interval}']
            feature_dict[f'ema5_ema20_ratio_{interval}'] = feature_dict[f'ema5_{interval}'] / feature_dict[f'ema20_{interval}']
            feature_dict[f'vol_change_momentum_{interval}'] = feature_dict[f'vol_change_{interval}'] * feature_dict[f'momentum_{interval}']

    # Combine all the new features with the original dataframe
    feature_df = pd.DataFrame(feature_dict, index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    return df

# Function to label breakout events based on price changes
def label_breakouts(df, min_price_change=0.005, max_price_drop=0.001, time_window=120):
    """
    Labels breakout events based on price changes within a specified time window.

    Parameters:
    - df (pd.DataFrame): The input dataframe with stock data.
    - min_price_change (float): Minimum percentage change to qualify as a breakout.
    - max_price_drop (float): Maximum allowed percentage drop to still consider it a breakout.
    - time_window (int): Number of future time steps to evaluate the breakout condition.

    Returns:
    - pd.DataFrame: The dataframe with a new 'breakout_type' column.
    """
    df['breakout_type'] = 0  # Initialize the 'breakout_type' column

    # Ensure necessary columns are present
    required_columns = ['close_1min']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Calculate future price changes over the time window
    future_prices = df['close_1min'].shift(-time_window)
    price_change = (future_prices - df['close_1min']) / df['close_1min']

    # Upward Breakout: Price increases by >= min_price_change without any significant drop
    upward_breakout = (price_change >= min_price_change) & (df['close_1min'] >= (1 - max_price_drop) * df['close_1min'])
    df.loc[upward_breakout, 'breakout_type'] = 1  # Label upward breakout

    # Downward Breakout: Price decreases by >= min_price_change without significant rise
    downward_breakout = (price_change <= -min_price_change) & (df['close_1min'] <= (1 + max_price_drop) * df['close_1min'])
    df.loc[downward_breakout, 'breakout_type'] = 2  # Label downward breakout

    return df

# Function to prepare LSTM data
def prepare_lstm_data(df, features, target, time_steps=1500):  # Increased time steps
    """
    Prepares sequential data for LSTM.

    Parameters:
    - df (pd.DataFrame): The input dataframe with features and target.
    - features (list): List of feature column names.
    - target (str): The target column name.
    - time_steps (int): Number of time steps for sequences.

    Returns:
    - X (np.ndarray): 3D array for LSTM input.
    - y (np.ndarray): 1D array of target labels.
    """
    X, y = [], []
    data_length = len(df)
    # Create sequences of data for the LSTM to use
    for i in range(time_steps, data_length):
        X.append(df[features].iloc[i-time_steps:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# Build the LSTM model
def build_lstm_model(input_shape, num_classes, units_1=256, units_2=128, units_3=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Constructs and compiles the LSTM neural network model.

    Parameters:
    - input_shape (tuple): Shape of the input data (time_steps, features).
    - num_classes (int): Number of output classes.
    - units_1 (int): Number of units in the first LSTM layer.
    - units_2 (int): Number of units in the second LSTM layer.
    - units_3 (int): Number of units in the third LSTM layer.
    - dropout_rate (float): Dropout rate for regularization.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - model (tf.keras.Model): The compiled LSTM model.
    """
    # Create LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=units_1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))  # Add dropout for regularization
    model.add(BatchNormalization())  # Normalize activations

    model.add(LSTM(units=units_2, return_sequences=True))  # Second LSTM layer
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(LSTM(units=units_3))  # Third LSTM layer
    model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation='relu', kernel_regularizer='l2'))  # Fully connected layer with L2 regularization
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax for multi-class classification
    
    opt = Adam(learning_rate=learning_rate)  # Use Adam optimizer
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model
    return model

def main():
    # Path to data file
    data_file = 'combined_data.txt'

    # Chunking parameters
    chunk_size = 5000
    overlap_size = 1000
    time_steps = 1500  # Increased time steps
    batch_size = 64

    features = None
    num_classes = 3  # Classes: 0 (No Breakout), 1 (Upward Breakout), 2 (Downward Breakout)
    scaler_lstm = StandardScaler()
    model_initialized = False

    all_y_true = []
    all_y_pred_lstm = []

    # Check if data file exists
    if not os.path.exists(data_file):
        logging.error(f"Data file '{data_file}' not found.")
        return

    try:
        chunk_iterator = pd.read_csv(data_file, sep='\t', chunksize=chunk_size - overlap_size, parse_dates=['timestamp'])
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        return

    chunk_number = 0
    prev_chunk_tail = None

    # Process data chunk by chunk
    for chunk in chunk_iterator:
        chunk_number += 1
        logging.info(f"Processing chunk {chunk_number}")

        if prev_chunk_tail is not None:
            chunk = pd.concat([prev_chunk_tail, chunk], ignore_index=True)

        prev_chunk_tail = chunk.iloc[-overlap_size:].copy()

        if 'timestamp' in chunk.columns:
            chunk = chunk.set_index('timestamp')
        else:
            logging.error("Column 'timestamp' not found in the data.")
            continue

        # Ensure all numerical columns are floats and convert to float32
        for col in chunk.columns:
            if chunk[col].dtype not in ['float64', 'float32']:
                try:
                    chunk[col] = chunk[col].astype(np.float32)
                except ValueError:
                    logging.warning(f"Column '{col}' could not be converted to float. Leaving as is.")

        # Perform feature engineering
        chunk = add_enhanced_features(chunk)

        # Label breakouts
        chunk = label_breakouts(chunk, min_price_change=0.005, max_price_drop=0.001, time_window=120)

        # Handle infinity or extremely large values
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)

        # Define features after processing the first chunk
        if features is None:
            features = [col for col in chunk.columns if col != 'breakout_type']
            logging.info(f"Features defined: {features}")

        # Check if there are enough data points after the time steps
        if len(chunk) <= time_steps:
            logging.info(f"Chunk {chunk_number} skipped due to insufficient data after time_steps.")
            continue

        # Prepare LSTM dataset
        chunk_data_lstm = chunk.copy()

        # Balance the dataset for LSTM
        breakout_indices_lstm = chunk_data_lstm[chunk_data_lstm['breakout_type'] > 0].index
        no_breakout_indices_lstm = chunk_data_lstm[chunk_data_lstm['breakout_type'] == 0].index

        num_breakouts_lstm = len(breakout_indices_lstm)
        num_no_breakouts_lstm = min(len(no_breakout_indices_lstm), num_breakouts_lstm * 10)

        if num_breakouts_lstm == 0:
            logging.info(f"Chunk {chunk_number} skipped due to no breakout instances.")
            continue

        # Randomly select 'No Breakout' instances
        try:
            selected_no_breakout_indices_lstm = np.random.choice(no_breakout_indices_lstm, size=num_no_breakouts_lstm, replace=False)
        except ValueError as e:
            logging.warning(f"Chunk {chunk_number} skipped due to insufficient 'No Breakout' instances: {e}")
            continue

        # Create balanced chunk for LSTM
        combined_indices_lstm = np.concatenate((breakout_indices_lstm, selected_no_breakout_indices_lstm))
        balanced_chunk_lstm = chunk_data_lstm.loc[combined_indices_lstm].sort_index()

        # Normalize the data for LSTM
        balanced_chunk_lstm[features] = scaler_lstm.fit_transform(balanced_chunk_lstm[features]).astype(np.float32)

        # Prepare LSTM input and output
        X_chunk_lstm, y_chunk_lstm = prepare_lstm_data(balanced_chunk_lstm, features, 'breakout_type', time_steps=time_steps)

        y_chunk_lstm = y_chunk_lstm.astype(int)

        # Skip if there's no data after preparation
        if len(X_chunk_lstm) == 0:
            logging.info(f"Chunk {chunk_number} skipped due to insufficient data after preparing sequences.")
            continue

        # One-hot encode the labels for LSTM
        y_chunk_categorical_lstm = to_categorical(y_chunk_lstm, num_classes=num_classes)

        # Class distribution
        class_distribution = Counter(y_chunk_lstm)

        # Split data into train and test sets
        try:
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                X_chunk_lstm, y_chunk_categorical_lstm, test_size=0.2, random_state=42, stratify=y_chunk_lstm)
        except ValueError as e:
            logging.warning(f"Chunk {chunk_number} skipped during train-test split: {e}")
            continue

        # Initialize and train LSTM model
        if not model_initialized:
            lstm_model = build_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), num_classes=num_classes)
            early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
            checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='loss', save_best_only=True, verbose=1)
            model_initialized = True
            logging.info("LSTM model initialized.")

        # Add class weights to prioritize breakout prediction
        class_weight = {0: 1, 1: 5, 2: 5}  # Higher weights for upward and downward breakouts

        # Train the model
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=batch_size, verbose=1,
                       callbacks=[early_stopping, checkpoint], class_weight=class_weight)
        logging.info(f"LSTM model trained on chunk {chunk_number}.")

        # Make predictions with LSTM
        y_pred_lstm_prob = lstm_model.predict(X_test_lstm)
        y_pred_lstm = np.argmax(y_pred_lstm_prob, axis=1)
        y_test_lstm_labels = np.argmax(y_test_lstm, axis=1)
        all_y_true.extend(y_test_lstm_labels)
        all_y_pred_lstm.extend(y_pred_lstm)

        logging.info(f"LSTM predictions made on chunk {chunk_number}.")

        # Clear variables to free memory
        del chunk, chunk_data_lstm, balanced_chunk_lstm, X_chunk_lstm, y_chunk_lstm, y_chunk_categorical_lstm
        del X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm
        gc.collect()

    # Evaluate the LSTM model
    if len(all_y_true) > 0:
        y_true = np.array(all_y_true)
        y_pred_lstm = np.array(all_y_pred_lstm)
        
        # Define class names
        class_names = ['No Breakout', 'Upward Breakout', 'Downward Breakout']
        
        # LSTM Model Evaluation
        logging.info("\nLSTM Model Evaluation:")
        lstm_report = classification_report(
            y_true, y_pred_lstm, labels=[0, 1, 2], target_names=class_names, zero_division=0)
        logging.info(f"LSTM Classification Report:\n{lstm_report}")
        
        lstm_conf_matrix = confusion_matrix(y_true, y_pred_lstm, labels=[0, 1, 2])
        logging.info(f"LSTM Confusion Matrix:\n{lstm_conf_matrix}")
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(lstm_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('LSTM Confusion Matrix')
        plt.show()
    else:
        logging.warning("No data collected for evaluation.")

if __name__ == "__main__":
    main()
