import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess  # To call gettingdata.py
import schedule  # To schedule periodic tasks
import threading  # To handle concurrent tasks
import time
import gc  # Garbage Collector for memory management

# ---------------------------- Configuration ----------------------------

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("dailytraining.log"),
        logging.StreamHandler()
    ]
)

# Suppress TensorFlow INFO and WARNING logs (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = filter out all
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Define global constants
DATA_FILE = 'data.txt'          # File updated by gettingdata.py
MODEL_PATH = 'best_lstm_model.h5'        # Path to save/load the LSTM model
PREDICTIONS_LOG = 'predictions_log.csv'  # Log file for predictions
TRAIN_ROWS = 1500                        # Number of rows for training
PREDICT_ROWS = 1001                      # Number of rows for prediction
BATCH_SIZE = 64                          # Batch size for training
NUM_CLASSES = 3                          # Number of output classes
RETRAIN_TIME = "23:59"                   # Daily retraining time (24-hour format)
SYMBOL = 'SPY'                            # Stock symbol to analyze

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
        
        # Ensure the headers match the expected number of columns
        if len(headers) != len(headers):  # Adjust this if you expect a certain number of columns
            logging.error(f"Unexpected number of columns found. Exiting...")
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

# ---------------------------- Model Functions ----------------------------

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
    model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
        if close_col in df.columns and open_col in df.columns:
            feature_dict[f'previous_close_{interval}'] = df[close_col].shift(1)
            feature_dict[f'price_change_{interval}'] = df[close_col] - df[open_col]
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
            
            # Interaction terms
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.volatility import BollingerBands

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
    df.fillna(0, inplace=True)  # Fill NaN values with 0 for consistency
    return df

def label_breakouts(df, min_price_change=0.005, time_window=120):
    """
    Labels breakout events based on price changes within a specified time window.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe with stock data.
    - min_price_change (float): Minimum percentage change to qualify as a breakout.
    - time_window (int): Number of future time steps to evaluate the breakout condition.
    
    Returns:
    - pd.DataFrame: The dataframe with a new 'breakout_type' column.
    """
    df['breakout_type'] = 0  # 0 for 'No Breakout'

    # Ensure necessary columns are present
    required_columns = ['close_1min']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    future_prices = df['close_1min'].shift(-time_window)
    price_change = (future_prices - df['close_1min']) / df['close_1min']

    upward_breakout = price_change >= min_price_change
    df.loc[upward_breakout, 'breakout_type'] = 1

    downward_breakout = price_change <= -min_price_change
    df.loc[downward_breakout, 'breakout_type'] = 2

    return df

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

    df_predict = add_enhanced_features(df_predict)
    df_predict = label_breakouts(df_predict)
    df_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_predict.fillna(0, inplace=True)

    if len(df_predict) < PREDICT_ROWS:
        logging.warning(f"Not enough data for prediction. Required: {PREDICT_ROWS}, Available: {len(df_predict)}")
        return predictions_log_df

    latest_data = df_predict[-PREDICT_ROWS:].copy()
    X_input, _ = prepare_lstm_data(latest_data, features, 'breakout_type', time_steps=1000)
    if len(X_input) == 0:
        logging.warning("Failed to prepare LSTM input for prediction.")
        return predictions_log_df

    X_input_scaled = scaler.transform(X_input.reshape(-1, len(features))).astype(np.float32)
    X_input_scaled = X_input_scaled.reshape(1, 1000, len(features))

    prediction_prob = model.predict(X_input_scaled)
    predicted_class = np.argmax(prediction_prob, axis=1)[0]
    confidence = np.max(prediction_prob) * 100
    current_time = latest_data.index[-1]

    logging.info(f"Prediction at {current_time}: {['No Breakout', 'Upward Breakout', 'Downward Breakout'][predicted_class]} ({confidence:.2f}%)")

    new_entry = pd.DataFrame({
        'timestamp': [current_time],
        'predicted_class': [predicted_class],
        'confidence': [confidence]
    })
    predictions_log_df = pd.concat([predictions_log_df, new_entry], ignore_index=True)

    predictions_log_df.to_csv(PREDICTIONS_LOG, index=False)
    logging.info(f"Prediction logged to '{PREDICTIONS_LOG}'.")

    gc.collect()

    return predictions_log_df

def train_model(model, scaler, features, df_train):
    """
    Trains the LSTM model on the provided dataset.
    
    Parameters:
    - model (tf.keras.Model): The trained LSTM model.
    - scaler (StandardScaler): The fitted scaler.
    - features (list): List of feature names.
    - df_train (pd.DataFrame): DataFrame containing training data.
    
    Returns:
    - None
    """
    target = 'breakout_type'
    X_train, y_train = prepare_lstm_data(df_train, features, target, time_steps=1000)
    if len(X_train) == 0:
        logging.error("Not enough data to prepare LSTM sequences for training.")
        return

    y_train = y_train.astype(int)
    y_train_categorical = to_categorical(y_train, num_classes=NUM_CLASSES)

    X_train_scaled = scaler.transform(X_train.reshape(-1, len(features))).astype(np.float32)
    X_train_scaled = X_train_scaled.reshape(X_train.shape[0], 1000, len(features))

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train_categorical, test_size=0.2, random_state=42, stratify=y_train
    )
    logging.info(f"Training data shape: {X_train_split.shape}, Validation data shape: {X_val.shape}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    model.fit(
        X_train_split, y_train_split,
        epochs=20,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping, checkpoint]
    )
    logging.info("Model training completed.")

    gc.collect()

# ---------------------------- Scheduler and Main ----------------------------

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
    schedule.every(1).minutes.do(lambda: make_prediction(model, scaler, features, predictions_log_df))

    schedule.every().day.at(RETRAIN_TIME).do(lambda: train_model(model, scaler, features, read_last_n_rows(DATA_FILE, TRAIN_ROWS)))

    logging.info("Scheduler started. Running scheduled tasks...")

    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)

    logging.info("Scheduler stopped.")

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

def main():
    df_initial = read_last_n_rows(DATA_FILE, TRAIN_ROWS)
    if df_initial.empty:
        logging.error("No data available for initial training. Exiting...")
        return

    df_initial = add_enhanced_features(df_initial)
    df_initial = label_breakouts(df_initial)
    df_initial.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_initial.fillna(0, inplace=True)

    target = 'breakout_type'
    features = [col for col in df_initial.columns if col != target]

    scaler = StandardScaler()
    X_initial, y_initial = prepare_lstm_data(df_initial, features, target, time_steps=1000)
    if len(X_initial) == 0:
        logging.error("Not enough data to prepare LSTM sequences for initial training.")
        return

    y_initial = y_initial.astype(int)
    y_initial_categorical = to_categorical(y_initial, num_classes=NUM_CLASSES)

    X_initial_scaled = scaler.fit_transform(X_initial.reshape(-1, len(features))).astype(np.float32)
    X_initial_scaled = X_initial_scaled.reshape(X_initial.shape[0], 1000, len(features))

    X_train, X_val, y_train, y_val = train_test_split(
        X_initial_scaled, y_initial_categorical, test_size=0.2, random_state=42, stratify=y_initial
    )
    logging.info(f"Initial Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    model = load_or_initialize_model(input_shape=(1000, len(features)), num_classes=NUM_CLASSES)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping, checkpoint]
    )
    logging.info("Initial model training completed.")

    gc.collect()

    if os.path.exists(PREDICTIONS_LOG):
        predictions_log_df = pd.read_csv(PREDICTIONS_LOG)
    else:
        predictions_log_df = pd.DataFrame(columns=['timestamp', 'predicted_class', 'confidence'])

    stop_event = threading.Event()

    scheduler_thread = threading.Thread(target=run_scheduler, args=(model, scaler, features, predictions_log_df, stop_event))
    scheduler_thread.start()

    listen_for_quit(stop_event)

    scheduler_thread.join()

if __name__ == "__main__":
    main()
