import concurrent.futures
import time
import schedule
import threading
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import pytz  # Added import for pytz

# Import your API key from apikey.py
import apikey

# ----------------------- Configuration -----------------------

# Polygon.io API key
api_key = apikey.API_KEY  # **Use your API key from apikey.py**

# Symbol to fetch data for
symbol = 'SPY'

# Define the intervals and their corresponding numerical values
interval_map = {
    '1min': '1',
    '5min': '5',
    '15min': '15',
    '30min': '30',
    '60min': '60'
}

# Define the order of intervals to assign suffixes
INTERVAL_ORDER = ['60min', '30min', '15min', '5min', '1min']
SUFFIX_MAP = {interval: ('' if idx == 0 else f'.{idx}') for idx, interval in enumerate(INTERVAL_ORDER)}

# Fetching parameters
CHUNK_DURATION = timedelta(days=60)             # 2 months
INITIAL_FETCH_DURATION = timedelta(days=5*365)  # 5 years

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("data_update.log"),
        logging.StreamHandler()
    ]
)

# Event to signal threads to stop
stop_event = threading.Event()

# ----------------------- Helper Functions -----------------------

def get_latest_timestamp(file_path='data.txt'):
    """
    Reads the last non-empty line from the file and extracts the timestamp.
    Assumes that the first column is 'timestamp'.
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, 2)  # Move to the end of the file
            file_size = f.tell()
            buffer_size = 1024
            if file_size == 0:
                return None  # Empty file
            elif file_size < buffer_size:
                buffer_size = file_size
            f.seek(-buffer_size, 2)
            last_bytes = f.read(buffer_size)
            last_str = last_bytes.decode('utf-8', errors='ignore')
            lines = last_str.strip().split('\n')
            # Iterate from last to first to find the last valid line
            for line in reversed(lines):
                if line.strip() and not line.startswith('timestamp'):
                    parts = line.split('\t')
                    if len(parts) > 0:
                        timestamp_str = parts[0]
                        try:
                            # Parse the timestamp and set timezone to UTC
                            latest_timestamp = pd.to_datetime(timestamp_str).tz_localize('UTC')
                            return latest_timestamp
                        except Exception as e:
                            logging.error(f"Error parsing timestamp '{timestamp_str}': {e}")
                            continue
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.error(f"Error reading latest timestamp from {file_path}: {e}")
        print(f"Error reading latest timestamp from {file_path}: {e}")
        return None

def fetch_data(symbol, interval, suffix, start_datetime, end_datetime, retries=3, timeout=10):
    """
    Fetches data from Polygon.io API for the specified symbol, interval, and datetime range.
    Returns a DataFrame with renamed columns to include interval suffix.
    """
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    # Convert datetime to Unix timestamps in milliseconds
    start_unix = int(start_datetime.timestamp() * 1000)
    end_unix = int(end_datetime.timestamp() * 1000)

    url = (f"{base_url}/{symbol}/range/{interval_map[interval]}/minute/"
           f"{start_unix}/{end_unix}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")

    logging.info(f"Fetching data for {interval} from {start_unix} to {end_unix}")
    print(f"Calling API: {url}")

    for attempt in range(1, retries + 1):
        if stop_event.is_set():
            logging.info(f"Stop event set. Exiting fetch_data for interval {interval}.")
            return pd.DataFrame()

        try:
            response = requests.get(url, timeout=timeout)
            logging.info(f"Attempt {attempt}: Response status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                logging.info(f"API returned {len(results)} results for interval {interval}.")
                print(f"Fetched {len(results)} rows for interval {interval}.")

                if results:
                    df = pd.DataFrame(results)
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    # Rename columns to include interval suffix where needed
                    df.rename(columns={
                        'v': f'volume_{interval}',
                        'vw': f'vw{suffix}',
                        'o': f'open_{interval}',
                        'c': f'close_{interval}',
                        'h': f'high_{interval}',
                        'l': f'low_{interval}',
                        't': f't{suffix}',
                        'n': f'n{suffix}'
                    }, inplace=True)

                    # Create 'vwap_<interval>' as a copy of 'vw{suffix}'
                    df[f'vwap_{interval}'] = df[f'vw{suffix}']

                    return df
                else:
                    logging.warning(f"No data returned for interval {interval}.")
                    print(f"No data returned for interval {interval}.")
                    return pd.DataFrame()
            elif response.status_code == 403:
                logging.error(f"403 Forbidden: {response.json().get('message')}")
                print(f"403 Forbidden: {response.json().get('message')}")
                break  # Don't retry on forbidden
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', '60'))
                logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                logging.error(f"Error fetching data: {response.status_code} - {response.text}")
                print(f"Error fetching data: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception on attempt {attempt} for interval {interval}: {e}")
            print(f"Request exception on attempt {attempt} for interval {interval}: {e}")

        if attempt < retries:
            wait_time = 2 ** attempt + (2 ** attempt) * 0.1  # Exponential backoff with jitter
            logging.info(f"Waiting for {wait_time:.2f} seconds before next retry...")
            print(f"Waiting for {wait_time:.2f} seconds before next retry...")
            time.sleep(wait_time)

    logging.error(f"Failed to fetch data for interval {interval} after {retries} attempts.")
    return pd.DataFrame()

def remove_duplicates(df):
    """
    Removes duplicate indices from the DataFrame.
    """
    if df.index.duplicated().any():
        logging.info(f"Removing {df.index.duplicated().sum()} duplicate entries.")
        df = df[~df.index.duplicated(keep='first')]
    return df

def forward_fill_data(df):
    """
    Forward-fills missing data in the DataFrame.
    """
    df = df.sort_index()
    for col in df.columns:
        df[col] = df[col].ffill()
    return df

def is_market_day():
    """
    Checks if today is a market day (Monday to Friday).
    """
    now_utc = datetime.now(pytz.utc)  # Adjusted to use UTC time
    return now_utc.weekday() < 5  # Monday is 0, Sunday is 6

def validate_data(df):
    """
    Validates the DataFrame to ensure data integrity.
    """
    # Check for missing values in critical columns
    critical_columns = [col for col in df.columns if 'vwap' in col or 'close' in col]
    if df[critical_columns].isnull().any().any():
        logging.warning("Missing values detected in critical columns.")
        print("Missing values detected in critical columns.")
    # Additional validation rules can be added here

def fetch_data_for_all_intervals(symbol, intervals, start_datetime, end_datetime):
    """
    Fetches data for all intervals for the specified date range and combines them into one DataFrame.
    """
    dataframes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(intervals)) as executor:
        futures = {
            executor.submit(fetch_data, symbol, interval, SUFFIX_MAP[interval], start_datetime, end_datetime): interval
            for interval in intervals
        }
        for future in concurrent.futures.as_completed(futures):
            interval = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    df = remove_duplicates(df)
                    dataframes.append(df)
                    logging.info(f"Data fetched for interval {interval}.")
                    print(f"Data fetched for interval {interval}.")
                else:
                    logging.warning(f"No data fetched for interval {interval}.")
                    print(f"No data fetched for interval {interval}.")
            except Exception as exc:
                logging.error(f'Generated an exception for interval {interval}: {exc}')
                print(f'Generated an exception for interval {interval}: {exc}')
    if dataframes:
        combined_df = combine_interval_data(dataframes)
        return combined_df
    else:
        return pd.DataFrame()

def combine_interval_data(dataframes):
    """
    Combines multiple DataFrames along columns, removes duplicates, and forward-fills missing data.
    """
    logging.info("Combining data from all intervals...")
    print("Combining data from all intervals...")
    combined_df = pd.concat(dataframes, axis=1)  # Concatenate along columns
    combined_df = remove_duplicates(combined_df)  # Ensure no duplicate index values
    combined_df = forward_fill_data(combined_df)

    # Define the desired column order based on your specified header
    desired_columns = [
        'timestamp',
        'volume_60min', 'vw', 'open_60min', 'close_60min', 'high_60min', 'low_60min', 't', 'n',
        'volume_30min', 'vw.1', 'open_30min', 'close_30min', 'high_30min', 'low_30min', 't.1', 'n.1',
        'volume_15min', 'vw.2', 'open_15min', 'close_15min', 'high_15min', 'low_15min', 't.2', 'n.2',
        'volume_5min', 'vw.3', 'open_5min', 'close_5min', 'high_5min', 'low_5min', 't.3', 'n.3',
        'volume_1min', 'vw.4', 'open_1min', 'close_1min', 'high_1min', 'low_1min', 't.4', 'n.4',
        'vwap_60min', 'vwap_30min', 'vwap_15min', 'vwap_5min', 'vwap_1min'
    ]

    # Reset index to have 'timestamp' as a column
    combined_df.reset_index(inplace=True)

    # Add any missing columns with NaN
    for col in desired_columns:
        if col not in combined_df.columns:
            combined_df[col] = pd.NA

    # Reorder columns
    combined_df = combined_df[desired_columns]

    # Set 'timestamp' back as index
    combined_df.set_index('timestamp', inplace=True)

    return combined_df

def fetch_and_append_latest_data(symbol):
    """
    Fetches the latest data for all intervals and appends to data.txt.
    """
    try:
        latest_timestamp = get_latest_timestamp('data.txt')
        if latest_timestamp is None:
            logging.error("data.txt not found or empty. Please run the initial data fetch first.")
            print("data.txt not found or empty. Please run the initial data fetch first.")
            return
        # Define the end_datetime as now -16 minutes in UTC
        end_datetime = datetime.now(pytz.utc) - timedelta(minutes=16)  # Adjusted to use UTC time
        # Define the start_datetime as the next minute after the latest_timestamp
        start_datetime = latest_timestamp + timedelta(minutes=1)
        # If start_datetime is after end_datetime, no new data to fetch
        if start_datetime >= end_datetime:
            logging.info("No new data to fetch.")
            print("No new data to fetch.")
            return
        # Fetch data for all intervals
        combined_df = fetch_data_for_all_intervals(symbol, INTERVAL_ORDER, start_datetime, end_datetime)
        if not combined_df.empty:
            # Append to data.txt
            combined_df.to_csv('data.txt', sep='\t', mode='a', header=False)
            logging.info(f"Fetched and appended latest data up to {combined_df.index.max()}.")
            print(f"Fetched and appended latest data up to {combined_df.index.max()}.")
        else:
            logging.warning("No new data fetched.")
            print("No new data fetched.")
    except Exception as e:
        logging.error(f"Error fetching latest data: {e}")
        print(f"Error fetching latest data: {e}")

def fetch_all_missing_data():
    """
    Fetches all missing data for all intervals and updates data.txt.
    """
    try:
        logging.info("Starting to fetch all missing data...")
        print("Starting to fetch all missing data...")
        latest_timestamp = get_latest_timestamp('data.txt')
        if latest_timestamp is None:
            # No existing data, fetch from INITIAL_FETCH_DURATION ago
            start_datetime = datetime.now(pytz.utc) - INITIAL_FETCH_DURATION  # Adjusted to use UTC time
        else:
            # Fetch from next minute after latest timestamp
            start_datetime = latest_timestamp + timedelta(minutes=1)
        # Define end_datetime as now -16 minutes in UTC
        end_datetime = datetime.now(pytz.utc) - timedelta(minutes=16)  # Adjusted to use UTC time
        if start_datetime >= end_datetime:
            logging.info("No new data to fetch.")
            print("No new data to fetch.")
            return
        # Fetch data in chunks
        current_start_datetime = start_datetime
        while current_start_datetime < end_datetime:
            current_end_datetime = min(current_start_datetime + CHUNK_DURATION, end_datetime)
            logging.info(f"Fetching data from {current_start_datetime} to {current_end_datetime}")
            print(f"Fetching data from {current_start_datetime} to {current_end_datetime}")
            combined_df = fetch_data_for_all_intervals(symbol, INTERVAL_ORDER, current_start_datetime, current_end_datetime)
            if not combined_df.empty:
                # Write header only if data.txt does not exist
                write_header = not os.path.exists('data.txt')
                combined_df.to_csv('data.txt', sep='\t', mode='a', header=write_header)
                logging.info(f"Fetched and saved data up to {combined_df.index.max()}.")
                print(f"Fetched and saved data up to {combined_df.index.max()}.")
                current_start_datetime = current_end_datetime + timedelta(minutes=1)
            else:
                logging.warning(f"No data fetched for period {current_start_datetime} to {current_end_datetime}.")
                print(f"No data fetched for period {current_start_datetime} to {current_end_datetime}.")
                break
    except Exception as e:
        logging.error(f"Error fetching all missing data: {e}")
        print(f"Error fetching all missing data: {e}")

def update_data_during_market_hours():
    """
    Fetches the latest data during market hours and appends to data.txt.
    """
    if not is_market_day():
        logging.info("Today is not a market day. Skipping data update.")
        print("Today is not a market day. Skipping data update.")
        return
    logging.info("Fetching latest data during market hours...")
    print("Fetching latest data during market hours...")
    fetch_and_append_latest_data(symbol)

def is_market_hours():
    """
    Checks if the current time in UTC is within market hours in UTC.
    """
    now_utc = datetime.now(pytz.utc)  # Adjusted to use UTC time
    now_time = now_utc.time()
    # Market hours in UTC for NYSE (Eastern Time): 13:30 to 20:00 UTC
    market_open = datetime.strptime("13:30", "%H:%M").time()
    market_close = datetime.strptime("20:00", "%H:%M").time()
    return market_open <= now_time <= market_close

def scheduled_task():
    """
    The task to run every minute during market hours.
    Logs each update attempt.
    """
    if is_market_hours():
        logging.info("Initiating scheduled data update.")
        print("Initiating scheduled data update.")
        start_update_time = datetime.now(pytz.utc)  # Adjusted to use UTC time
        update_data_during_market_hours()
        end_update_time = datetime.now(pytz.utc)  # Adjusted to use UTC time
        elapsed = (end_update_time - start_update_time).total_seconds()
        logging.info(f"Scheduled data update completed in {elapsed} seconds.")
        print(f"Scheduled data update completed in {elapsed} seconds.")
    else:
        logging.info("Market is closed. No update performed.")
        print("Market is closed. No update performed.")

def run_scheduler():
    """
    Runs the scheduler in a separate thread.
    """
    # Schedule fetch_all_missing_data to run at 08:00 AM UTC from Monday to Friday
    schedule.every().monday.at("08:00").do(fetch_all_missing_data)
    schedule.every().tuesday.at("08:00").do(fetch_all_missing_data)
    schedule.every().wednesday.at("08:00").do(fetch_all_missing_data)
    schedule.every().thursday.at("08:00").do(fetch_all_missing_data)
    schedule.every().friday.at("08:00").do(fetch_all_missing_data)

    # Schedule the scheduled_task to run every minute
    schedule.every().minute.do(scheduled_task)

    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)  # Check every second for pending tasks

def quit_gracefully():
    """
    Function to gracefully shutdown the scheduler.
    """
    logging.info("Quit signal received. Shutting down gracefully...")
    print("Quit signal received. Shutting down gracefully...")
    stop_event.set()

def quit_listener():
    """
    Listens for 'quit' or 'exit' commands to gracefully shutdown the program.
    """
    while not stop_event.is_set():
        user_input = input("Type 'quit' or 'exit' to stop the scheduler: ").strip().lower()
        if user_input in ['quit', 'exit']:
            quit_gracefully()
            break

def start_scheduler():
    """
    Starts the scheduler and quit listener on separate threads.
    """
    # Start the scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler)  # Removed daemon=True
    scheduler_thread.start()

    # Start the quit listener thread
    listener_thread = threading.Thread(target=quit_listener)  # Removed daemon=True
    listener_thread.start()

    # Wait for both threads to finish
    scheduler_thread.join()
    listener_thread.join()
    logging.info("Scheduler has stopped.")
    print("Scheduler has stopped.")

if __name__ == "__main__":
    # On program start, fetch missing data immediately
    fetch_all_missing_data()

    # Start the scheduler and quit listener
    start_scheduler()
