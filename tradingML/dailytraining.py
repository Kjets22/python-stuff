import pandas as pd
import numpy as np
import requests
import datetime
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import schedule  # For scheduling tasks
import threading  # For running scheduled tasks in the background
import joblib  # For loading the scaler and feature names

# Import technical analysis indicators
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# ---------------------------- Configuration ----------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Global Variables
API_KEY = 'LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg'  
  # Replace with your actual Polygon.io API key
SYMBOL = 'SPY'  # Stock symbol to analyze
TIME_INTERVAL = 1  # Time interval in minutes for data fetching
TIME_STEPS = 1000  # Number of time steps for model input
MODEL_PATH = 'best_lstm_model.h5'  # Path to your pre-trained model
FEATURE_NAMES_FILE = 'feature_names.txt'  # File containing feature names
SCALER_FILE = 'scaler.joblib'  # File containing the fitted scaler
DATA_DIR = 'data/'  # Directory to store data and reports
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize global data storage
real_time_data = pd.DataFrame()
predictions_log = pd.DataFrame()

# Initialize the stop event for graceful shutdown
stop_event = threading.Event()

# ---------------------------- Helper Functions ----------------------------


interval_map = {
    '1min': '1',
    '5min': '5',
    '15min': '15',
    '30min': '30',
    '60min': '60'
}

def add_enhanced_features(df):
    """
    Adds a variety of technical indicators and interaction terms to the dataframe.
    Ensures consistency between training and real-time data.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe containing stock data.
    
    Returns:
    - pd.DataFrame: The dataframe with new engineered features.
    """
    feature_dict = {}
    intervals = ['1min', '5min', '15min', '30min', '60min']

    for interval in intervals:
        close_col = f'close_{interval}'
        if close_col in df.columns:
            # Basic Features
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

            # Technical Indicators using 'ta' library
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

            # Interaction Terms
            feature_dict[f'ma5_ma20_ratio_{interval}'] = feature_dict[f'ma5_{interval}'] / feature_dict[f'ma20_{interval}']
            feature_dict[f'ema5_ema20_ratio_{interval}'] = feature_dict[f'ema5_{interval}'] / feature_dict[f'ema20_{interval}']
            feature_dict[f'vol_change_momentum_{interval}'] = feature_dict[f'vol_change_{interval}'] * feature_dict[f'momentum_{interval}']

    # Concatenate all the new features to the original dataframe
    feature_df = pd.DataFrame(feature_dict, index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    df.fillna(0, inplace=True)  # Fill NaN values with 0 for consistency
    return df

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
    upward_breakout = (price_change >= min_price_change) & (df['close_1min'] >= (1 - max_price_drop) * df['close_1min'])

    # Assign Upward Breakout Label
    df.loc[upward_breakout, 'breakout_type'] = 1

    # Downward Breakout: Price decreases by >= min_price_change without any rise >= max_price_drop
    downward_breakout = (price_change <= -min_price_change) & (df['close_1min'] <= (1 + max_price_drop) * df['close_1min'])

    # Assign Downward Breakout Label
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
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_pretrained_model():
    """
    Loads the pre-trained LSTM model.

    Returns:
    - model (tf.keras.Model): The loaded LSTM model.
    """
    try:
        model = load_model(MODEL_PATH)
        logging.info("Pre-trained model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

def forward_fill_data(df):
    """
    Forward-fills all missing values for each column in the DataFrame.
    """
    for col in df.columns:
        df[col] = df[col].ffill()
    df.fillna(0, inplace=True)
    return df

def fetch_real_time_data(initial_fetch=False):
    """
    Fetches real-time data from Polygon.io and appends it to the global real_time_data DataFrame.
    Aggregates data into multiple intervals to match training data.

    Parameters:
    - initial_fetch (bool): If True, fetches enough historical data to populate TIME_STEPS.
    """
    global real_time_data
    logging.info("Fetching real-time data...")

    # Get current UTC time
    now = datetime.datetime.utcnow()

    # Adjust for any API delay (e.g., 15 minutes)
    delay_minutes = 15  # Adjust if necessary

    if initial_fetch:
        # Fetch enough historical data to have TIME_STEPS data points
        total_minutes_needed = TIME_STEPS * 10  # Fetch more to account for aggregation
        from_time = now - datetime.timedelta(minutes=delay_minutes + total_minutes_needed)
    else:
        from_time = now - datetime.timedelta(minutes=delay_minutes + TIME_INTERVAL)

    to_time = now - datetime.timedelta(minutes=delay_minutes)

    # Convert to UNIX timestamps in milliseconds
    from_timestamp = int(from_time.timestamp() * 1000)
    to_timestamp = int(to_time.timestamp() * 1000)

    # Base URL for fetching data
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    url = f"{base_url}/{SYMBOL}/range/{interval_map['1min']}/minute/{from_timestamp}/{to_timestamp}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={
                'o': 'open_1min',
                'h': 'high_1min',
                'l': 'low_1min',
                'c': 'close_1min',
                'v': 'volume_1min'
            }, inplace=True)
            df = df[['open_1min', 'high_1min', 'low_1min', 'close_1min', 'volume_1min']]

            # Aggregate data to 5min, 15min, 30min, and 60min intervals
            df_aggregated = pd.DataFrame(index=df.index)

            # Define resampling rules with 'min' instead of 'T'
            resample_rules = {
                '5min': '5min',
                '15min': '15min',
                '30min': '30min',
                '60min': '60min'
            }

            for interval, rule in resample_rules.items():
                df_aggregated[f'open_{interval}'] = df['open_1min'].resample(rule).first()
                df_aggregated[f'high_{interval}'] = df['high_1min'].resample(rule).max()
                df_aggregated[f'low_{interval}'] = df['low_1min'].resample(rule).min()
                df_aggregated[f'close_{interval}'] = df['close_1min'].resample(rule).last()
                df_aggregated[f'volume_{interval}'] = df['volume_1min'].resample(rule).sum()

            # Drop NaN values that might result from resampling
            df_aggregated.dropna(inplace=True)

            # Combine 1min and aggregated data
            df_combined = pd.concat([df, df_aggregated], axis=1)

            # Forward-fill missing data
            df_combined = forward_fill_data(df_combined)

            # Append to real_time_data
            real_time_data = pd.concat([real_time_data, df_combined]).drop_duplicates()
            logging.info(f"Fetched and appended {len(df_combined)} new data points.")
        else:
            logging.warning("No new data fetched.")
    except Exception as e:
        logging.error(f"Error fetching real-time data: {e}")

def make_predictions(model, scaler):
    """
    Uses the latest TIME_STEPS data points to make predictions.

    Parameters:
    - model (tf.keras.Model): The pre-trained LSTM model.
    - scaler (StandardScaler): The scaler fitted on the training data.
    """
    global real_time_data, predictions_log
    logging.info("Making predictions...")

    if len(real_time_data) < TIME_STEPS:
        logging.warning("Not enough data to make predictions.")
        return

    # Prepare the data
    data = real_time_data.copy()

    # Apply feature engineering
    data = add_enhanced_features(data)

    # Handle missing values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Load required features
    try:
        with open(FEATURE_NAMES_FILE, 'r') as f:
            required_features = [line.strip() for line in f]
    except FileNotFoundError:
        logging.error(f"Feature list file '{FEATURE_NAMES_FILE}' not found. Ensure it exists.")
        return

    # Check for missing features and add them with default value 0
    missing_features = set(required_features) - set(data.columns)
    if missing_features:
        logging.warning(f"Missing features detected: {missing_features}. Adding them with default value 0.")
        for feature in missing_features:
            data[feature] = 0

    # Ensure data has all required features
    data = data[required_features]

    # Log the number of features
    logging.info(f"Number of features after feature engineering: {len(required_features)}")

    # Log missing features
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
    else:
        logging.info("All required features are present.")

    # Scale the features using the scaler
    try:
        data[required_features] = scaler.transform(data[required_features]).astype(np.float32)
    except Exception as e:
        logging.error(f"Error during feature scaling: {e}")
        return

    # Prepare the latest sequence
    X_input = data[required_features].iloc[-TIME_STEPS:].values.reshape(1, TIME_STEPS, -1)

    # Make prediction
    try:
        prediction = model.predict(X_input)
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        return

    predicted_class = np.argmax(prediction, axis=1)[0]
    class_labels = ['No Breakout', 'Upward Breakout', 'Downward Breakout']
    confidence = np.max(prediction) * 100
    current_time = data.index[-1]

    # Log the prediction
    predictions_log = predictions_log.append({
        'timestamp': current_time,
        'predicted_class': predicted_class,
        'confidence': confidence
    }, ignore_index=True)

    # If a breakout is predicted, display it
    if predicted_class != 0:
        logging.info(f"Breakout Predicted at {current_time}: {class_labels[predicted_class]} with confidence {confidence:.2f}%")

def retrain_model():
    """
    Retrains the model using the day's data and updates the 'best_lstm_model.h5' file.
    """
    global real_time_data
    logging.info("Retraining the model with today's data...")

    if len(real_time_data) < TIME_STEPS:
        logging.warning("Not enough data to retrain the model.")
        return

    # Prepare the data
    data = real_time_data.copy()

    # Apply feature engineering and labeling
    data = add_enhanced_features(data)
    data = label_breakouts(data)

    # Handle missing values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Load required features
    try:
        with open(FEATURE_NAMES_FILE, 'r') as f:
            required_features = [line.strip() for line in f]
    except FileNotFoundError:
        logging.error(f"Feature list file '{FEATURE_NAMES_FILE}' not found.")
        return

    target = 'breakout_type'

    # Ensure all required features are present
    missing_features = set(required_features) - set(data.columns)
    if missing_features:
        logging.warning(f"Missing features detected during retraining: {missing_features}. Adding them with default value 0.")
        for feature in missing_features:
            data[feature] = 0

    # Ensure data has all required features
    data = data[required_features + [target]]

    # Scale the features using the same scaler
    try:
        scaler = joblib.load(SCALER_FILE)
        data[required_features] = scaler.transform(data[required_features]).astype(np.float32)
    except FileNotFoundError:
        logging.error(f"Scaler file '{SCALER_FILE}' not found. Ensure it exists.")
        return
    except Exception as e:
        logging.error(f"Error during feature scaling in retraining: {e}")
        return

    # Prepare data for LSTM
    X_data, y_data = prepare_lstm_data(data, required_features, target, time_steps=TIME_STEPS)
    y_data = y_data.astype(int)
    y_categorical = to_categorical(y_data, num_classes=3)

    # Check if there are enough samples for training
    if len(X_data) < 10:
        logging.warning("Not enough data to retrain the model.")
        return

    # Split data into training and validation sets
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_categorical, test_size=0.2, random_state=42, stratify=y_data
        )
        logging.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    except ValueError as e:
        logging.warning(f"Train-test split failed during retraining: {e}")
        return

    # Build a new model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, num_classes=3)

    # Calculate class weights to handle class imbalance
    y_integers = np.argmax(y_train, axis=1)
    class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weights_dict = dict(enumerate(class_weights_array))

    # Implement Early Stopping and Model Checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    try:
        model.fit(
            X_train, y_train,
            epochs=10,  # Adjust as needed
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            class_weight=class_weights_dict
        )
        logging.info("Model retrained and saved successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    # Clear the day's data
    real_time_data.drop(real_time_data.index, inplace=True)
    predictions_log.drop(predictions_log.index, inplace=True)
    logging.info("Cleared the day's data for the next trading day.")

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on test data.

    Parameters:
    - model (tf.keras.Model): The trained LSTM model.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): Test labels (one-hot encoded).
    """
    logging.info("Evaluating the model...")
    try:
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

        class_names = ['No Breakout', 'Upward Breakout', 'Downward Breakout']
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        logging.info(f"Classification Report:\n{report}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")

def schedule_tasks(model, scaler):
    """
    Schedules tasks for fetching data, making predictions, and retraining the model.
    """
    # Schedule data fetching and prediction during market hours
    schedule.every(TIME_INTERVAL).minutes.do(lambda: fetch_and_predict(model, scaler))

    # Schedule model retraining after market close
    schedule.every().day.at("16:10").do(retrain_model)  # Adjust the time as needed

    # Start the scheduler and listener in separate threads
    threading.Thread(target=run_scheduler, daemon=True).start()
    threading.Thread(target=listen_for_quit, daemon=True).start()

def fetch_and_predict(model, scaler):
    """
    Combines data fetching and prediction.
    """
    if stop_event.is_set():
        logging.info("Stopping fetch_and_predict due to stop event.")
        return schedule.CancelJob  # This will cancel the scheduled job

    current_time = datetime.datetime.now().time()
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)

    if market_open <= current_time <= market_close:
        fetch_real_time_data()
        make_predictions(model, scaler)
    else:
        logging.info("Market is closed. Skipping data fetch and prediction.")

def run_scheduler():
    """
    Runs the scheduler indefinitely.
    """
    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)
    logging.info("Scheduler stopped.")

def listen_for_quit():
    """
    Listens for the 'quit' or 'exit' command to stop the program.
    """
    logging.info("Type 'quit' or 'exit' to stop the program.")
    while not stop_event.is_set():
        try:
            user_input = input()
            if user_input.strip().lower() in ['quit', 'exit']:
                logging.info("Quit command received. Stopping the program...")
                stop_event.set()
                schedule.clear()  # Clear all scheduled jobs
                break
        except EOFError:
            # Handle end-of-file (e.g., if input stream is closed)
            break

def main():
    # Load the pre-trained model
    model = load_pretrained_model()

    # Load the scaler
    try:
        scaler = joblib.load(SCALER_FILE)
        logging.info("Scaler loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Scaler file '{SCALER_FILE}' not found. Ensure it exists.")
        return
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        return

    # Load the feature names
    try:
        with open(FEATURE_NAMES_FILE, 'r') as f:
            required_features = [line.strip() for line in f]
        logging.info(f"Loaded {len(required_features)} feature names from '{FEATURE_NAMES_FILE}'.")
    except FileNotFoundError:
        logging.error(f"Feature names file '{FEATURE_NAMES_FILE}' not found. Ensure it exists.")
        return
    except Exception as e:
        logging.error(f"Error loading feature names: {e}")
        return

    # Fetch initial historical data
    fetch_real_time_data(initial_fetch=True)

    # Apply feature engineering
    real_time_data_processed = add_enhanced_features(real_time_data)

    # Handle missing values
    real_time_data_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    real_time_data_processed.fillna(0, inplace=True)

    # Check for missing features and add them with default value 0
    missing_features = set(required_features) - set(real_time_data_processed.columns)
    if missing_features:
        logging.warning(f"Missing features detected in initial data: {missing_features}. Adding them with default value 0.")
        for feature in missing_features:
            real_time_data_processed[feature] = 0

    # Ensure data has all required features
    real_time_data_processed = real_time_data_processed[required_features]

    # Scale the features
    try:
        real_time_data_processed[required_features] = scaler.transform(real_time_data_processed[required_features]).astype(np.float32)
    except Exception as e:
        logging.error(f"Error during feature scaling of initial data: {e}")
        return

    # Update the global real_time_data with processed data
    real_time_data.update(real_time_data_processed)

    # Start scheduling tasks
    schedule_tasks(model, scaler)

    # Keep the main thread alive until stop_event is set
    while not stop_event.is_set():
        time.sleep(1)
    logging.info("Program has been terminated.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Stopping the program...")
        stop_event.set()
        schedule.clear()
