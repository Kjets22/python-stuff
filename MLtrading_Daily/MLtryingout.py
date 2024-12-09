
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import schedule
import threading
import time
import gc
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------------- Configuration ----------------------------

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("combined_training.log"),
        logging.StreamHandler()
    ]
)

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = filter out all
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Enable GPU growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Enabled memory growth for {len(physical_devices)} GPU(s).")
    except Exception as e:
        logging.error(f"Could not set memory growth: {e}")
else:
    logging.warning("No GPU found. Running on CPU.")

# Define global constants
DATA_FILE = 'data.txt'                    # File updated by gettingdata.py
MODEL_PATH = 'best_lstm_model_tryingout.keras'      # Path to save/load the LSTM model
PREDICTIONS_LOG = 'predictions_log.csv'   # Log file for predictions
TRAIN_CHUNK_SIZE = 3000                   # Number of rows per training chunk
TRAIN_OVERLAP_SIZE = 1500                 # Number of overlapping rows between training chunks
TIME_STEPS = 1500                         # Number of time steps for LSTM
PREDICT_ROWS = TIME_STEPS + 1             # Number of rows for prediction
BATCH_SIZE = 64                           # Batch size for training
NUM_CLASSES = 3                           # Number of output classes
RETRAIN_TIME = "23:59"                    # Daily retraining time (24-hour format)
SYMBOL = 'SPY'                             # Stock symbol to analyze
 
# ---------------------------- Helper Functions ----------------------------

def read_header(file_path):
    """
    Reads the header (first line) of the CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - headers (list): List of column names.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            headers = header_line.split('\t')  # The file is tab-separated
            return headers
    except Exception as e:
        logging.error(f"Error reading header from '{file_path}': {e}")
        return []

def read_overlapping_chunks(file_path, chunk_size, overlap_size):
    """
    Generator function to read overlapping chunks from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - chunk_size (int): Number of rows per chunk.
    - overlap_size (int): Number of overlapping rows between chunks.

    Yields:
    - pd.DataFrame: Next chunk of data with overlap.
    """
    try:
        headers = read_header(file_path)
        if not headers:
            logging.error("No headers found. Exiting...")
            return

        # Initialize the iterator
        reader = pd.read_csv(
            file_path,
            sep='\t',
            chunksize=chunk_size,
            iterator=True,
            skiprows=1,  # Skip header
            header=None,
            names=headers,
            parse_dates=['timestamp'] if 'timestamp' in headers else False
        )

        prev_chunk = pd.DataFrame()
        for chunk in reader:
            if 'timestamp' in chunk.columns:
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
                chunk.set_index('timestamp', inplace=True)
            if not prev_chunk.empty:
                # Combine the last overlap_size rows from previous chunk with the current chunk
                combined_chunk = pd.concat([prev_chunk.tail(overlap_size), chunk], axis=0)
            else:
                combined_chunk = chunk
            yield combined_chunk
            prev_chunk = chunk
    except Exception as e:
        logging.error(f"Error reading overlapping chunks from '{file_path}': {e}")
        return

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
        open_col = f'open_{interval}'
        volume_col = f'volume_{interval}'
        high_col = f'high_{interval}'
        low_col = f'low_{interval}'
        
        if close_col in df.columns and open_col in df.columns and volume_col in df.columns and high_col in df.columns and low_col in df.columns:
            feature_dict[f'previous_close_{interval}'] = df[close_col].shift(1)
            feature_dict[f'price_change_{interval}'] = df[close_col] - df[open_col]
            feature_dict[f'ma5_{interval}'] = df[close_col].rolling(window=5, min_periods=1).mean()
            feature_dict[f'ma10_{interval}'] = df[close_col].rolling(window=10, min_periods=1).mean()
            feature_dict[f'ma20_{interval}'] = df[close_col].rolling(window=20, min_periods=1).mean()
            feature_dict[f'vol_change_{interval}'] = df[volume_col].pct_change(fill_method=None).fillna(0)
            feature_dict[f'high_low_diff_{interval}'] = df[high_col] - df[low_col]
            feature_dict[f'open_close_diff_{interval}'] = df[open_col] - df[close_col]
            feature_dict[f'ema5_{interval}'] = df[close_col].ewm(span=5, adjust=False).mean()
            feature_dict[f'ema20_{interval}'] = df[close_col].ewm(span=20, adjust=False).mean()
            feature_dict[f'momentum_{interval}'] = df[close_col] - df[close_col].shift(4).fillna(0)
            feature_dict[f'volatility_{interval}'] = df[close_col].rolling(window=5, min_periods=1).std()
            feature_dict[f'roc_{interval}'] = df[close_col].pct_change(periods=10, fill_method=None)
            feature_dict[f'ema12_{interval}'] = df[close_col].ewm(span=12, adjust=False).mean()
            feature_dict[f'ema26_{interval}'] = df[close_col].ewm(span=26, adjust=False).mean()
            feature_dict[f'macd_{interval}'] = feature_dict[f'ema12_{interval}'] - feature_dict[f'ema26_{interval}']
            
            # New indicators using 'ta' library
            try:
                from ta.momentum import RSIIndicator, StochasticOscillator
                from ta.volatility import BollingerBands

                # RSI
                rsi = RSIIndicator(df[close_col], window=14)
                feature_dict[f'rsi_{interval}'] = rsi.rsi()
            except Exception as e:
                logging.warning(f"RSI Indicator failed for {interval}: {e}")
                feature_dict[f'rsi_{interval}'] = np.nan
            try:
                bb_indicator = BollingerBands(df[close_col], window=20, window_dev=2)
                feature_dict[f'bb_high_{interval}'] = bb_indicator.bollinger_hband()
                feature_dict[f'bb_low_{interval}'] = bb_indicator.bollinger_lband()
            except Exception as e:
                logging.warning(f"Bollinger Bands failed for {interval}: {e}")
                feature_dict[f'bb_high_{interval}'] = np.nan
                feature_dict[f'bb_low_{interval}'] = np.nan
            try:
                stoch_indicator = StochasticOscillator(
                    high=df[high_col],
                    low=df[low_col],
                    close=df[close_col],
                    window=14,
                    smooth_window=3
                )
                feature_dict[f'stoch_k_{interval}'] = stoch_indicator.stoch()
                feature_dict[f'stoch_d_{interval}'] = stoch_indicator.stoch_signal()
            except Exception as e:
                logging.warning(f"Stochastic Oscillator failed for {interval}: {e}")
                feature_dict[f'stoch_k_{interval}'] = np.nan
                feature_dict[f'stoch_d_{interval}'] = np.nan
                
            # Interaction terms
            feature_dict[f'ma5_ma20_ratio_{interval}'] = feature_dict[f'ma5_{interval}'] / (feature_dict[f'ma20_{interval}'] + 1e-8)
            feature_dict[f'ema5_ema20_ratio_{interval}'] = feature_dict[f'ema5_{interval}'] / (feature_dict[f'ema20_{interval}'] + 1e-8)
            feature_dict[f'vol_change_momentum_{interval}'] = feature_dict[f'vol_change_{interval}'] * feature_dict[f'momentum_{interval}']

    # Concatenate all the new features to the original dataframe
    feature_df = pd.DataFrame(feature_dict, index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    df.fillna(0, inplace=True)  # Fill NaN values with 0 for consistency
    return df

def label_breakouts(df, min_price_change=0.005, max_opposite_change=0.001, time_window=120):
    """
    Labels breakout events based on price changes within a specified time window and limits
    on the opposite direction change.

    Parameters:
    - df (pd.DataFrame): The input dataframe with stock data.
    - min_price_change (float): Minimum percentage change to qualify as a breakout.
    - max_opposite_change (float): Maximum allowable percentage change in the opposite direction.
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

    # Iterate over all time steps within the time_window
    for step in range(1, time_window + 1):
        # Calculate the price change at this step
        future_prices = df['close_1min'].shift(-step)
        price_change = (future_prices - df['close_1min']) / df['close_1min']

        # Calculate the minimum price within the time window so far
        min_price = df['close_1min'].rolling(window=step, min_periods=1).min().shift(-step)
        max_price = df['close_1min'].rolling(window=step, min_periods=1).max().shift(-step)

        # Upward Breakout: Price increases by >= min_price_change and does not drop below max_opposite_change
        upward_breakout = (price_change >= min_price_change) & \
                          ((min_price - df['close_1min']) / df['close_1min'] >= -max_opposite_change)

        # Downward Breakout: Price decreases by <= -min_price_change and does not rise above max_opposite_change
        downward_breakout = (price_change <= -min_price_change) & \
                            ((max_price - df['close_1min']) / df['close_1min'] <= max_opposite_change)

        # Assign labels
        df.loc[upward_breakout, 'breakout_type'] = 1  # Label as Upward Breakout
        df.loc[downward_breakout, 'breakout_type'] = 2  # Label as Downward Breakout

    return df
## old breakout code did not check for anytime can have price change
# def label_breakouts(df, min_price_change=0.005, time_window=120):
#     """
#     Labels breakout events based on price changes within a specified time window.

#     Parameters:
#     - df (pd.DataFrame): The input dataframe with stock data.
#     - min_price_change (float): Minimum percentage change to qualify as a breakout.
#     - time_window (int): Number of future time steps to evaluate the breakout condition.

#     Returns:
#     - pd.DataFrame: The dataframe with a new 'breakout_type' column.
#     """
#     # Initialize the 'breakout_type' column with a default value
#     df['breakout_type'] = 0  # 0 for 'No Breakout'

#     # Ensure necessary columns are present
#     required_columns = ['close_1min']
#     for col in required_columns:
#         if col not in df.columns:
#             df[col] = np.nan

#     # Calculate future price changes over the time window
#     future_prices = df['close_1min'].shift(-time_window)
#     price_change = (future_prices - df['close_1min']) / df['close_1min']

#     # Upward Breakout: Price increases by >= min_price_change
#     upward_breakout = price_change >= min_price_change

#     # Assign Upward Breakout Label
#     df.loc[upward_breakout, 'breakout_type'] = 1

#     # Downward Breakout: Price decreases by >= min_price_change
#     downward_breakout = price_change <= -min_price_change

#     # Assign Downward Breakout Label
#     df.loc[downward_breakout, 'breakout_type'] = 2

#     return df

def prepare_lstm_data(df, features, target, time_steps=1500):
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

def get_class_weights(y, num_classes):
    """
    Computes class weights to handle class imbalance.

    Parameters:
    - y (np.ndarray): Array of target labels.
    - num_classes (int): Total number of classes.

    Returns:
    - dict: Class weights dictionary with keys from 0 to num_classes - 1.
    """
    # Initialize class weights with default value
    class_weight_dict = {i: 1.0 for i in range(num_classes)}
    # Compute class weights only for classes present in y
    classes_in_y = np.unique(y)
    if len(classes_in_y) < 2:
        logging.warning(f"Only one class present in y: {classes_in_y}. Using default class weights.")
        return class_weight_dict
    weights = compute_class_weight('balanced', classes=classes_in_y, y=y)
    for cls, weight in zip(classes_in_y, weights):
        class_weight_dict[int(cls)] = weight
    return class_weight_dict

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
    model.add(Input(shape=input_shape))  # Explicit Input layer
    model.add(LSTM(units=units_1, return_sequences=True))
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

def load_or_initialize_model(input_shape, num_classes):
    """
    Loads an existing model or initializes a new one if not found.

    Parameters:
    - input_shape (tuple): Shape of the input data.
    - num_classes (int): Number of output classes.

    Returns:
    - model (tf.keras.Model): Loaded or newly initialized model.
    """
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            logging.info(f"Loaded existing model from '{MODEL_PATH}'.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
    # Initialize a new model if loading fails
    model = build_lstm_model(input_shape, num_classes)
    logging.info("Initialized a new LSTM model.")
    return model

# ---------------------------- Prediction and Training Functions ----------------------------

def make_prediction(model, scaler, features, predictions_log_df):
    """
    Makes a prediction based on the latest data and logs it.

    Parameters:
    - model (tf.keras.Model): The trained LSTM model.
    - scaler (StandardScaler): The fitted scaler.
    - features (list): List of feature names.
    - predictions_log_df (pd.DataFrame): DataFrame to log predictions.

    Returns:
    - predictions_log_df (pd.DataFrame): Updated predictions log.
    """
    df_predict = read_last_n_rows(DATA_FILE, PREDICT_ROWS)
    if df_predict.empty:
        logging.warning("No data available for prediction.")
        return predictions_log_df
    print("part1 complete")
    df_predict = add_enhanced_features(df_predict)
    df_predict = label_breakouts(df_predict)
    df_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_predict.fillna(0, inplace=True)

    if len(df_predict) < PREDICT_ROWS:
        logging.warning(f"Not enough data for prediction. Required: {PREDICT_ROWS}, Available: {len(df_predict)}")
        return predictions_log_df
    print("part2 complete")

    latest_data = df_predict[-PREDICT_ROWS:].copy()
    X_input, _ = prepare_lstm_data(latest_data, features, 'breakout_type', time_steps=TIME_STEPS)
    if len(X_input) == 0:
        logging.warning("Failed to prepare LSTM input for prediction.")
        return predictions_log_df

    print("part3 complete")
    # Reshape and scale
    X_input_scaled = scaler.transform(X_input.reshape(-1, len(features))).astype(np.float32)
    X_input_scaled = X_input_scaled.reshape(1, TIME_STEPS, len(features))

    # Make prediction
    prediction_prob = model.predict(X_input_scaled)
    predicted_class = np.argmax(prediction_prob, axis=1)[0]
    confidence = np.max(prediction_prob) * 100
    current_time = latest_data.index[-1]
    
    print("part4 complete")
    logging.info(f"Prediction at {current_time}: {['No Breakout', 'Upward Breakout', 'Downward Breakout'][predicted_class]} ({confidence:.2f}%)")

    # Log prediction
    new_entry = pd.DataFrame({
        'timestamp': [current_time],
        'predicted_class': [predicted_class],
        'confidence': [confidence]
    })
    predictions_log_df = pd.concat([predictions_log_df, new_entry], ignore_index=True)

    # Save predictions log
    predictions_log_df.to_csv(PREDICTIONS_LOG, index=False)
    logging.info(f"Prediction logged to '{PREDICTIONS_LOG}'.")

    gc.collect()

    return predictions_log_df

def train_model_on_chunk(model, scaler, features, df_train):
    """
    Trains the LSTM model on a single data chunk.

    Parameters:
    - model (tf.keras.Model): The trained LSTM model.
    - scaler (StandardScaler): The fitted scaler.
    - features (list): List of feature names.
    - df_train (pd.DataFrame): DataFrame containing training data.

    Returns:
    - None
    """
    target = 'breakout_type'
    X_train, y_train = prepare_lstm_data(df_train, features, target, time_steps=TIME_STEPS)
    if len(X_train) == 0:
        logging.error("Not enough data to prepare LSTM sequences for training.")
        return

    y_train = y_train.astype(int)
    y_train_categorical = to_categorical(y_train, num_classes=NUM_CLASSES)

    # Compute class weights
    class_weights = get_class_weights(y_train, NUM_CLASSES)

    # Check for missing classes
    missing_classes = set(range(NUM_CLASSES)) - set(np.unique(y_train))
    if missing_classes:
        logging.warning(f"Missing classes in y_train: {missing_classes}. Assigning default weights to missing classes.")

    # Scale the data
    X_train_scaled = scaler.transform(X_train.reshape(-1, len(features))).astype(np.float32)
    X_train_scaled = X_train_scaled.reshape(X_train.shape[0], TIME_STEPS, len(features))

    # Split into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train_categorical, test_size=0.2, random_state=42
    )
    logging.info(f"Training on chunk: {X_train_split.shape[0]} samples, Validation: {X_val.shape[0]} samples.")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model on this chunk with class weights
    model.fit(
        X_train_split, y_train_split,
        epochs=10,  # Adjust epochs as needed
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weights  # Apply class weights here
    )
    logging.info("Model training on chunk completed.")

    # Evaluate the model on validation data
    evaluate_model(model, X_val, y_val)

    # Save the model after each chunk
    model.save(MODEL_PATH)
    logging.info(f"Model saved to '{MODEL_PATH}' after chunk.")

    gc.collect()


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model using validation data and logs the results.
    """
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Define class names
    class_names = ['No Breakout', 'Upward Breakout', 'Downward Breakout']

    # Classification Report
    report = classification_report(
        y_true, y_pred, 
        labels=[0, 1, 2], 
        target_names=class_names,
        zero_division=0
    )
    logging.info(f"LSTM Classification Report:\n{report}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    logging.info(f"LSTM Confusion Matrix:\n{conf_matrix}")

    # Define a function to plot the confusion matrix
    def plot_confusion_matrix():
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('LSTM Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.close()
            logging.info("Confusion matrix plotted and saved successfully.")
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")

    # Start plotting in a separate thread
    plot_thread = threading.Thread(target=plot_confusion_matrix)
    plot_thread.start()
    
def read_last_n_rows(file_path, n):
    """
    Efficiently reads the last n rows from a CSV file without loading the entire file into memory.

    Parameters:
    - file_path (str): Path to the CSV file.
    - n (int): Number of rows to read from the end.

    Returns:
    - df (pd.DataFrame): DataFrame containing the last n rows.
    """
    try:
        headers = read_header(file_path)
        if not headers:
            logging.error("No headers found. Exiting...")
            return pd.DataFrame()

        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            buffer = bytearray()
            pointer = filesize - 1
            lines = []
            while pointer >= 0 and len(lines) < n:
                f.seek(pointer)
                byte = f.read(1)
                if byte == b'\n':
                    if buffer:
                        line = buffer[::-1].decode('utf-8', errors='ignore')
                        lines.append(line)
                        buffer = bytearray()
                else:
                    buffer.extend(byte)
                pointer -= 1
            if buffer:
                line = buffer[::-1].decode('utf-8', errors='ignore')
                lines.append(line)
            # Reverse to have the lines in correct order
            lines = lines[::-1]
            # Take the last n lines
            last_n_lines = lines[-n:]
            # Convert to DataFrame, using tab separation
            df = pd.DataFrame([x.split('\t') for x in last_n_lines], columns=headers)

            # Convert 'timestamp' column to datetime if it exists
            if 'timestamp' in headers:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.set_index('timestamp', inplace=True)

            # Convert all other columns to numeric, coerce errors to NaN, then fill NaN with 0
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            logging.info(f"Loaded the last {n} rows from '{file_path}' with {len(headers)} columns.")
            return df
    except Exception as e:
        logging.error(f"Error reading last {n} rows from '{file_path}': {e}")
        return pd.DataFrame()

def call_gettingdata():
    """
    Calls the gettingdata.py script to fetch and update data.
    """
    try:
        subprocess.run(['python', 'gettingdata.py'], check=True)
        logging.info("Successfully executed gettingdata.py.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing gettingdata.py: {e}")


def run_scheduler(model, scaler, features, predictions_log_df, stop_event):
    """
    Runs the scheduler to handle periodic tasks.

    Parameters:
    - model (tf.keras.Model): The trained LSTM model.
    - scaler (StandardScaler): The fitted scaler.
    - features (list): List of feature names.
    - predictions_log_df (pd.DataFrame): DataFrame to log predictions.
    - stop_event (threading.Event): Event to signal the script to stop.

    Returns:
    - None
    """
    print("in schedular")
    # Schedule prediction every minute
    schedule.every(1).minutes.do(lambda: make_prediction(model, scaler, features, predictions_log_df))

    # Schedule end-of-day summary at RETRAIN_TIME
    schedule.every().day.at(RETRAIN_TIME).do(lambda: end_of_day_summary(predictions_log_df))

    # Schedule retraining daily at RETRAIN_TIME
    schedule.every().day.at(RETRAIN_TIME).do(lambda: retrain_model(model, scaler, features))

    logging.info("Scheduler started. Running scheduled tasks...")

    while not stop_event.is_set():
        schedule.run_pending()
        logging.debug("scheduler is running..")
        time.sleep(1)

    logging.info("Scheduler stopped.")

def end_of_day_summary(predictions_log_df):
    """
    Generates and displays a summary of all predictions made during the day.
    """
    if predictions_log_df.empty:
        logging.warning("No predictions to summarize for today.")
        return

    logging.info("Generating end-of-day summary of predictions...")

    # Calculate summary statistics
    total_predictions = len(predictions_log_df)
    class_counts = predictions_log_df['predicted_class'].value_counts().to_dict()
    confidence_mean = predictions_log_df['confidence'].mean()
    confidence_std = predictions_log_df['confidence'].std()

    logging.info(f"End-of-Day Summary:")
    logging.info(f"Total Predictions: {total_predictions}")
    logging.info(f"Predicted Classes: {class_counts}")
    logging.info(f"Average Confidence: {confidence_mean:.2f}%")
    logging.info(f"Confidence Standard Deviation: {confidence_std:.2f}%")

    # Define a function to generate plots
    def generate_plots():
        try:
            # Plot confidence distribution
            plt.figure(figsize=(8, 6))
            sns.histplot(predictions_log_df['confidence'], bins=20, kde=True)
            plt.xlabel('Confidence (%)')
            plt.ylabel('Frequency')
            plt.title('End-of-Day Prediction Confidence Distribution')
            plt.savefig('end_of_day_confidence_distribution.png')
            plt.close()

            # Plot prediction counts per class
            plt.figure(figsize=(8, 6))
            sns.countplot(x='predicted_class', data=predictions_log_df, palette='viridis')
            plt.xlabel('Predicted Class')
            plt.ylabel('Count')
            plt.title('End-of-Day Prediction Counts')
            plt.xticks(ticks=[0,1,2], labels=['No Breakout', 'Upward Breakout', 'Downward Breakout'])
            plt.savefig('end_of_day_prediction_counts.png')
            plt.close()

            logging.info("Plots generated and saved successfully.")
        except Exception as e:
            logging.error(f"Error generating plots: {e}")

    # Start plotting in a separate thread
    plot_thread = threading.Thread(target=generate_plots)
    plot_thread.start()

    # Proceed with clearing the predictions log
    # predictions_log_df.drop(predictions_log_df.index, inplace=True)
    # predictions_log_df.to_csv(PREDICTIONS_LOG, index=False)
    logging.info("End-of-day summary completed and predictions log cleared.")
    
def retrain_model(model, scaler, features):
    """
    Retrains the model with the latest data.

    Parameters:
    - model (tf.keras.Model): The trained LSTM model.
    - scaler (StandardScaler): The fitted scaler.
    - features (list): List of feature names.

    Returns:
    - None
    """
    logging.info("Retraining the model with the latest data...")
    # Update data by calling gettingdata.py
    call_gettingdata()

    # Read the latest chunk for retraining
    df_train = read_last_n_rows(DATA_FILE, TRAIN_CHUNK_SIZE)
    if df_train.empty:
        logging.error("No data available for retraining.")
        return

    df_train = add_enhanced_features(df_train)
    df_train = label_breakouts(df_train)
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train.fillna(0, inplace=True)
    
    # Re-fit the scaler on new training data
    scaler.fit(df_train[features])
    logging.info("Scaler re-fitted on new training data.")
    
    # Train on the new chunk
    train_model_on_chunk(model, scaler, features, df_train)
    model = load_model(MODEL_PATH)
        
    logging.info("Reloaded the updated model after retraining.")

    
    # with open('scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)
    # with open('scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)
    logging.info("Reloaded the updated scaler after retraining.")

    
def listen_for_quit(stop_event):
    """
    Listens for the 'quit' or 'exit' command to gracefully terminate the script.

    Parameters:
    - stop_event (threading.Event): Event to signal the script to stop.

    Returns:
    - None
    """
    logging.info("Type 'quit' or 'exit' to stop the program.")
    while not stop_event.is_set():
        try:
            user_input = input()
            if user_input.strip().lower() in ['quit', 'exit']:
                logging.info("Shutdown command received. Stopping the program...")
                stop_event.set()
                break
        except EOFError:
            logging.info("EOF detected. Stopping the program...")
            stop_event.set()
            break
        except Exception as e:
            logging.error(f"Error reading input: {e}")

# ---------------------------- Main Function ----------------------------

def main():
    """
    Main function to orchestrate training, prediction, and scheduling.
    """
    # Initialize variables
    features = []
    scaler = None

    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)

    if not model_exists:
        logging.info("No existing trained model found. Starting initial training...")

        # Initialize the model after determining the number of features
        # Read the entire data in overlapping chunks for initial training
        chunk_generator = read_overlapping_chunks(DATA_FILE, TRAIN_CHUNK_SIZE, TRAIN_OVERLAP_SIZE)
        model = None
        scaler = StandardScaler()
        for idx, chunk in enumerate(chunk_generator):
            logging.info(f"Processing chunk {idx + 1} for training.")

            # Feature Engineering
            chunk = add_enhanced_features(chunk)
            chunk = label_breakouts(chunk)

            # Define features if not already defined
            if not features:
                # Exclude 'breakout_type' and ensure all are numeric
                features = [col for col in chunk.columns if col != 'breakout_type' and pd.api.types.is_numeric_dtype(chunk[col])]
                logging.info(f"Defined features: {features}")

            # Remove rows with invalid values in features
            invalid_rows = chunk[~np.isfinite(chunk[features]).all(axis=1)]
            if not invalid_rows.empty:
                logging.warning(f"Found {len(invalid_rows)} invalid rows in training chunk {idx + 1}. Removing them.")
                chunk = chunk[np.isfinite(chunk[features]).all(axis=1)]
            else:
                logging.info(f"All rows in chunk {idx + 1} are valid for scaling.")

            # Proceed with scaling using valid data
            scaler.fit(chunk[features])
            chunk_scaled = scaler.transform(chunk[features])

            # Prepare LSTM data
            # Include 'breakout_type' in the DataFrame
            df_scaled = pd.DataFrame(chunk_scaled, columns=features, index=chunk.index)
            df_final = pd.concat([df_scaled, chunk['breakout_type']], axis=1)

            X, y = prepare_lstm_data(df_final, features, 'breakout_type', time_steps=TIME_STEPS)
            if len(X) == 0:
                logging.warning(f"Not enough data in chunk {idx + 1} to create sequences. Skipping.")
                continue

            y = y.astype(int)
            y_categorical = to_categorical(y, num_classes=NUM_CLASSES)

            # Initialize model if first chunk
            if model is None:
                model = build_lstm_model(input_shape=(TIME_STEPS, len(features)), num_classes=NUM_CLASSES)
                logging.info("Initialized new LSTM model for training.")

            # Compute class weights
            class_weights = get_class_weights(y, NUM_CLASSES)

            # Check for missing classes
            missing_classes = set(range(NUM_CLASSES)) - set(np.unique(y))
            if missing_classes:
                logging.warning(f"Missing classes in y: {missing_classes}. Assigning default weights to missing classes.")

            # Reshape X to match expected input shape
            X_train_scaled = X.astype(np.float32)

            # Split into training and validation sets
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train_scaled, y_categorical, test_size=0.2, random_state=42
            )
            logging.info(f"Training on chunk {idx + 1}: {X_train_split.shape[0]} samples, Validation: {X_val.shape[0]} samples.")

            # Define callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

            # Train the model on this chunk with class weights
            model.fit(
                X_train_split, y_train_split,
                epochs=10,  # Adjust epochs as needed
                batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                verbose=1,
                callbacks=[early_stopping, checkpoint],
                class_weight=class_weights  # Apply class weights here
            )
            logging.info(f"Model training on chunk {idx + 1} completed.")

            # Evaluate the model on validation data
            evaluate_model(model, X_val, y_val)

            # Save the model after each chunk
            model.save(MODEL_PATH)
            logging.info(f"Model saved to '{MODEL_PATH}' after chunk {idx + 1}.")

            gc.collect()

        logging.info("Initial training on all chunks completed.")
    else:
        logging.info(f"Trained model found at '{MODEL_PATH}'. Loading the model for predictions.")
        # To define features, read a sample of data
        df_sample = read_last_n_rows(DATA_FILE, TRAIN_OVERLAP_SIZE)
        if df_sample.empty:
            logging.error("No data available to define features. Exiting...")
            return
        df_sample = add_enhanced_features(df_sample)
        df_sample = label_breakouts(df_sample)
        # Exclude 'breakout_type' and ensure all are numeric
        features = [col for col in df_sample.columns if col != 'breakout_type' and pd.api.types.is_numeric_dtype(df_sample[col])]

        # Remove rows with invalid values in features
        invalid_rows = df_sample[~np.isfinite(df_sample[features]).all(axis=1)]
        if not invalid_rows.empty:
            logging.warning(f"Found {len(invalid_rows)} invalid rows in sample data. Removing them.")
            df_sample = df_sample[np.isfinite(df_sample[features]).all(axis=1)]
        else:
            logging.info("All rows in sample data are valid for scaling.")

        scaler = StandardScaler()
        scaler.fit(df_sample[features])

        # Load the model
        model = load_or_initialize_model(input_shape=(TIME_STEPS, len(features)), num_classes=NUM_CLASSES)

    # Load or initialize predictions log
    if os.path.exists(PREDICTIONS_LOG):
        try:
            predictions_log_df = pd.read_csv(PREDICTIONS_LOG)
            logging.info(f"Loaded existing predictions log from '{PREDICTIONS_LOG}'.")
        except Exception as e:
            logging.error(f"Error loading predictions log: {e}")
            predictions_log_df = pd.DataFrame(columns=['timestamp', 'predicted_class', 'confidence'])
    else:
        predictions_log_df = pd.DataFrame(columns=['timestamp', 'predicted_class', 'confidence'])
        logging.info(f"Initialized new predictions log at '{PREDICTIONS_LOG}'.")

    # Start scheduler for periodic tasks
    stop_event = threading.Event()

    scheduler_thread = threading.Thread(target=run_scheduler, args=(model, scaler, features, predictions_log_df, stop_event))
    scheduler_thread.start()

    # Start listening for quit commands
    listen_for_quit(stop_event)

    scheduler_thread.join()

if __name__ == "__main__":
    main()
