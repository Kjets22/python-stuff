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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Delete existing .h5 files to prevent loading previous weights
if os.path.exists('best_lstm_model.h5'):
    os.remove('best_lstm_model.h5')
    logging.info("Existing LSTM model weights deleted.")

# Feature engineering with additional indicators
def add_enhanced_features(df):
    """
    Adds a variety of technical indicators and interaction terms to the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing stock data.

    Returns:
    - pd.DataFrame: The dataframe with new engineered features.
    """
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
            
            # New indicators using 'ta' library
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.volatility import BollingerBands

            # Handle cases where there are not enough data points
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
                stoch_indicator = StochasticOscillator(
                    high=df[f'high_{interval}'],
                    low=df[f'low_{interval}'],
                    close=df[close_col],
                    window=14,
                    smooth_window=3
                )
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

    # Upward Breakout: Price increases by >= min_price_change without any drop >= max_price_drop
    upward_breakout = (price_change >= min_price_change) & \
                      (df['close_1min'] >= (1 - max_price_drop) * df['close_1min'])

    # Assign Upward Breakout Label
    df.loc[upward_breakout, 'breakout_type'] = 1

    # Downward Breakout: Price decreases by >= min_price_change without any rise >= max_price_drop
    downward_breakout = (price_change <= -min_price_change) & \
                        (df['close_1min'] <= (1 + max_price_drop) * df['close_1min'])

    # Assign Downward Breakout Label
    df.loc[downward_breakout, 'breakout_type'] = 2

    return df

# Prepare data for LSTM
def prepare_lstm_data(df, features, target, time_steps=1000):
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
    for i in range(time_steps, data_length):
        X.append(df[features].iloc[i-time_steps:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# Build the LSTM model
def build_lstm_model(input_shape, num_classes, units_1=128, units_2=64, units_3=32, dropout_rate=0.3, learning_rate=0.001):
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
    model = Sequential()
    model.add(LSTM(units=units_1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units=units_3))
    model.add(Dropout(dropout_rate))
    # Adjusted Dense Layer to match the number of classes
    model.add(Dense(num_classes, activation='relu', kernel_regularizer='l2'))
    # Output Layer with softmax activation for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Define the sequence length for LSTM
    sequence_length = 1000  # Replaces both time_steps and overlap_size

    # Define chunk size and overlap
    chunk_size = 5000
    overlap_size = sequence_length  # Set overlap size equal to sequence_length

    time_steps_lstm = sequence_length  # Use sequence_length for LSTM time_steps

    batch_size = 64  # Increased batch size from 32 to 64
    features = None
    num_classes = 3  # Classes: 0 (No Breakout), 1 (Upward Breakout), 2 (Downward Breakout)
    scaler_lstm = StandardScaler()
    model_initialized = False

    # Initialize variables to collect evaluation metrics
    all_y_true = []
    all_y_pred_lstm = []

    # File path to your combined data
    data_file = 'combined_data.txt'  # Update this path as needed

    # Check if data file exists
    if not os.path.exists(data_file):
        logging.error(f"Data file '{data_file}' not found.")
        return

    # Read and process data in chunks
    try:
        chunk_iterator = pd.read_csv(data_file, sep='\t', chunksize=chunk_size - overlap_size, parse_dates=['timestamp'])
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        return

    chunk_number = 0
    prev_chunk_tail = None

    for chunk in chunk_iterator:
        chunk_number += 1
        logging.info(f"Processing chunk {chunk_number}")

        # If there is a previous chunk's tail, concatenate it
        if prev_chunk_tail is not None:
            chunk = pd.concat([prev_chunk_tail, chunk], ignore_index=True)
            logging.debug(f"Chunk {chunk_number} concatenated with previous tail.")

        # Keep the last 'overlap_size' data points for the next chunk
        prev_chunk_tail = chunk.iloc[-overlap_size:].copy()

        # Set 'timestamp' as the index
        if 'timestamp' in chunk.columns:
            chunk = chunk.set_index('timestamp')
        else:
            logging.error("Column 'timestamp' not found in the data.")
            continue

        # Ensure all numerical columns are floats
        for col in chunk.columns:
            if chunk[col].dtype not in ['float64', 'float32']:
                try:
                    chunk[col] = chunk[col].astype(float)
                    logging.debug(f"Column '{col}' converted to float.")
                except ValueError:
                    # Handle columns that cannot be converted to float
                    logging.warning(f"Column '{col}' could not be converted to float. Leaving as is.")

        # Perform feature engineering
        chunk = add_enhanced_features(chunk)
        logging.debug(f"Feature engineering completed for chunk {chunk_number}.")

        # Label breakouts
        chunk = label_breakouts(chunk, min_price_change=0.005, max_price_drop=0.001, time_window=120)
        logging.debug(f"Breakout labeling completed for chunk {chunk_number}.")

        # Handle infinity or extremely large values
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)
        logging.debug(f"Handled infinity and NaN values for chunk {chunk_number}.")

        # Define features after processing the first chunk
        if features is None:
            features = [col for col in chunk.columns if col != 'breakout_type']
            logging.info(f"Features defined: {features}")

        # Check if we have enough data points after time_steps
        if len(chunk) <= time_steps_lstm:
            logging.info(f"Chunk {chunk_number} skipped due to insufficient data after time_steps.")
            continue

        # Prepare the dataset for LSTM
        chunk_data_lstm = chunk.copy()

        # Balance the dataset for LSTM
        breakout_indices_lstm = chunk_data_lstm[chunk_data_lstm['breakout_type'] > 0].index
        no_breakout_indices_lstm = chunk_data_lstm[chunk_data_lstm['breakout_type'] == 0].index

        num_breakouts_lstm = len(breakout_indices_lstm)
        num_no_breakouts_lstm = min(len(no_breakout_indices_lstm), num_breakouts_lstm * 10)

        # Skip the chunk if there are no breakouts
        if num_breakouts_lstm == 0:
            logging.info(f"Chunk {chunk_number} skipped due to no breakout instances.")
            continue

        # Randomly select 'No Breakout' instances
        try:
            selected_no_breakout_indices_lstm = np.random.choice(no_breakout_indices_lstm, size=num_no_breakouts_lstm, replace=False)
            logging.debug(f"Selected {num_no_breakouts_lstm} 'No Breakout' instances for LSTM in chunk {chunk_number}.")
        except ValueError as e:
            logging.warning(f"Chunk {chunk_number} skipped due to insufficient 'No Breakout' instances: {e}")
            continue

        # Combine indices and create balanced chunk for LSTM
        combined_indices_lstm = np.concatenate((breakout_indices_lstm, selected_no_breakout_indices_lstm))
        balanced_chunk_lstm = chunk_data_lstm.loc[combined_indices_lstm].sort_index()
        logging.debug(f"Balanced LSTM chunk created for chunk {chunk_number} with {len(balanced_chunk_lstm)} samples.")

        # Normalize the data for LSTM
        balanced_chunk_lstm[features] = scaler_lstm.fit_transform(balanced_chunk_lstm[features])
        logging.debug(f"Data normalization completed for LSTM in chunk {chunk_number}.")

        # Prepare data for LSTM
        X_chunk_lstm, y_chunk_lstm = prepare_lstm_data(balanced_chunk_lstm, features, 'breakout_type', time_steps=time_steps_lstm)
        logging.debug(f"LSTM data prepared for chunk {chunk_number}.")

        # Ensure y_chunk is integer
        y_chunk_lstm = y_chunk_lstm.astype(int)

        # Skip if there's no data after preparation
        if len(X_chunk_lstm) == 0:
            logging.info(f"Chunk {chunk_number} skipped due to insufficient data after preparing sequences for LSTM.")
            continue

        # One-hot encode the labels for LSTM
        y_chunk_categorical_lstm = to_categorical(y_chunk_lstm, num_classes=num_classes)

        # Check the class distribution in y_chunk before splitting
        class_distribution = Counter(y_chunk_lstm)
        logging.debug(f"Class distribution for chunk {chunk_number}: {class_distribution}")

        # Ensure each class has at least 2 instances for stratified splitting
        if min(class_distribution.values()) < 2:
            logging.warning(f"Chunk {chunk_number}: Insufficient samples in one of the classes. Skipping stratified split.")
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                X_chunk_lstm, y_chunk_categorical_lstm, test_size=0.2, random_state=42
            )
        else:
            # Stratified splitting if there are at least 2 instances in each class
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                X_chunk_lstm, y_chunk_categorical_lstm, test_size=0.2, random_state=42, stratify=y_chunk_lstm
            )
            logging.debug(f"Stratified train-test split completed for chunk {chunk_number}.")

        # Initialize and train LSTM model
        if not model_initialized:
            lstm_model = build_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), num_classes=num_classes)
            # Implement Early Stopping and Model Checkpointing
            early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
            checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='loss', save_best_only=True, verbose=1)
            model_initialized = True
            logging.info("LSTM model initialized.")

        lstm_model.fit(
            X_train_lstm, y_train_lstm,
            epochs=10,  # Increased epochs with early stopping
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping, checkpoint]
        )
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
        logging.info(f"Memory cleared after processing chunk {chunk_number}.")

    # Evaluate the LSTM model
    if len(all_y_true) > 0:
        y_true = np.array(all_y_true)
        y_pred_lstm = np.array(all_y_pred_lstm)
        
        # Define class names
        class_names = ['No Breakout', 'Upward Breakout', 'Downward Breakout']
        
        # LSTM Model Evaluation
        logging.info("\nLSTM Model Evaluation:")
        lstm_report = classification_report(
            y_true, y_pred_lstm, 
            labels=[0, 1, 2], 
            target_names=class_names,
            zero_division=0
        )
        logging.info(f"LSTM Classification Report:\n{lstm_report}")
        
        lstm_conf_matrix = confusion_matrix(y_true, y_pred_lstm, labels=[0, 1, 2])
        logging.info(f"LSTM Confusion Matrix:\n{lstm_conf_matrix}")
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(lstm_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('LSTM Confusion Matrix')
        plt.show()
    else:
        logging.warning("No data collected for evaluation.")

if __name__ == "__main__":
    main()
