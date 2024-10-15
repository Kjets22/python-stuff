
import concurrent.futures
import time
import schedule
import threading
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo  # Requires Python 3.9+

# ----------------------- Configuration -----------------------

# Polygon.io API key
api_key = 'LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg'  # **Replace with your actual API key**

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
INITIAL_FETCH_DURATION = timedelta(days=5*365) # 5 years

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
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
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
    for col in df.columns:
        df[col] = df[col].ffill()
    return df


def is_market_day():
    """
    Checks if today is a market day (Monday to Friday).
    """
    now_eastern = datetime.now(ZoneInfo("US/Eastern"))
    return now_eastern.weekday() < 5  # Monday is 0, Sunday is 6


def fetch_latest_data(symbol, interval, suffix):
    """
    Fetches the latest data point for the specified interval and updates the data.txt file.
    This function accounts for a 16-minute delay in data availability.
    """
    try:
        # Read the latest timestamp from data.txt
        try:
            combined_df = pd.read_csv('data.txt', sep='\t', index_col='timestamp', parse_dates=True)
            # Ensure the index is timezone-aware in UTC
            if combined_df.index.tz is None:
                combined_df.index = combined_df.index.tz_localize('UTC')
            latest_timestamp = combined_df.index.max()
        except FileNotFoundError:
            logging.error("data.txt not found. Please run the initial data fetch first.")
            print("data.txt not found. Please run the initial data fetch first.")
            return

        # Define the end_datetime as now -16 minutes
        end_datetime = datetime.now(timezone.utc) - timedelta(minutes=16)
        # Define the start_datetime as the next minute after the latest_timestamp
        start_datetime = latest_timestamp + timedelta(minutes=1)

        # If start_datetime is after end_datetime, no new data to fetch
        if start_datetime >= end_datetime:
            logging.info(f"No new data to fetch for interval {interval}.")
            print(f"No new data to fetch for interval {interval}.")
            return

        # Fetch data from start_datetime to end_datetime
        df = fetch_data(symbol, interval, suffix, start_datetime, end_datetime)
        if not df.empty:
            df = remove_duplicates(df)
            combined_df = pd.concat([combined_df, df], axis=0)
            combined_df = remove_duplicates(combined_df)
            combined_df = forward_fill_data(combined_df)
            # Save the updated DataFrame
            combined_df.reset_index(inplace=True)
            combined_df.to_csv('data.txt', sep='\t', index=True, header=True)
            logging.info(f"Fetched and updated latest data for interval {interval} at {df.index.max()}.")
            print(f"Fetched and updated latest data for interval {interval} at {df.index.max()}.")
        else:
            logging.warning(f"No new data fetched for interval {interval}.")
            print(f"No new data fetched for interval {interval}.")

    except Exception as e:
        logging.error(f"Error fetching latest data for interval {interval}: {e}")
        print(f"Error fetching latest data for interval {interval}: {e}")


def update_combined_data(combined_df, interval, suffix):
    """
    Updates the combined DataFrame with new data for the specified interval.
    """
    logging.info(f"Checking for missing data for interval: {interval}...")
    print(f"Checking for missing data for interval: {interval}...")
    latest_date_combined = combined_df.index.max().astimezone(timezone.utc)
    current_end_date = datetime.now(timezone.utc)

    if latest_date_combined < current_end_date:
        logging.info(f"Data is missing from {latest_date_combined} to {current_end_date} for {interval}. Fetching missing data...")
        print(f"Data is missing from {latest_date_combined} to {current_end_date} for {interval}. Fetching missing data...")
        new_data_df = fetch_missing_data(
            symbol,
            interval,
            suffix,
            latest_date_combined,
            current_end_date
        )
        if not new_data_df.empty:
            logging.info(f"Fetched {len(new_data_df)} rows for interval {interval}.")
            print(f"Fetched {len(new_data_df)} rows for interval {interval}.")
            new_data_df = remove_duplicates(new_data_df)
            combined_df = pd.concat([combined_df, new_data_df], axis=0)
            combined_df = remove_duplicates(combined_df)
            combined_df = forward_fill_data(combined_df)
        else:
            logging.warning(f"No new data fetched for interval {interval}.")
            print(f"No new data fetched for interval {interval}.")
    else:
        logging.info(f"No missing data for interval {interval}.")
        print(f"No missing data for interval {interval}.")

    return combined_df


def fetch_missing_data(symbol, interval, suffix, start_datetime, end_datetime):
    """
    Fetches all missing data for a given interval between start_datetime and end_datetime in 60-day chunks.
    Datetimes should be timezone-aware in UTC.
    """
    all_data = []
    current_end_datetime = end_datetime
    current_start_datetime = current_end_datetime - CHUNK_DURATION

    while current_end_datetime > start_datetime:
        # Adjust if current_start_datetime is before start_datetime
        if current_start_datetime < start_datetime:
            current_start_datetime = start_datetime

        logging.info(f"Fetching {interval} data from {current_start_datetime} to {current_end_datetime}")
        print(f"Fetching {interval} data from {current_start_datetime} to {current_end_datetime}")

        df = fetch_data(symbol, interval, suffix, current_start_datetime, current_end_datetime)
        if not df.empty:
            df = remove_duplicates(df)  # Remove duplicates right after fetching
            all_data.append(df)
            # Update the end datetime to fetch the next batch
            current_end_datetime = current_start_datetime - timedelta(seconds=1)
            current_start_datetime = current_end_datetime - CHUNK_DURATION
            logging.info(f"Latest datetime reached for {interval}: {df.index.min()}")
            print(f"Latest datetime reached for {interval}: {df.index.min()}")
        else:
            logging.warning(f"No data fetched for interval {interval}. Check API response.")
            print(f"No data fetched for interval {interval}.")
            break

    if all_data:
        combined = pd.concat(all_data)
        logging.info(f"Total fetched rows for interval {interval}: {len(combined)}")
        print(f"Total fetched rows for interval {interval}: {len(combined)}")
        return combined
    else:
        logging.warning(f"No data was fetched for interval {interval}.")
        print(f"No data was fetched for interval {interval}.")
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
    return combined_df


def fetch_all_missing_data():
    """
    Fetches all missing data for all intervals and updates the data.txt file.
    """
    try:
        logging.info("Starting to fetch all missing data...")
        print("Starting to fetch all missing data...")

        try:
            combined_df = pd.read_csv('data.txt', sep='\t', index_col='timestamp', parse_dates=True)
            # Ensure the index is timezone-aware in UTC
            if combined_df.index.tz is None:
                combined_df.index = combined_df.index.tz_localize('UTC')
            logging.info(f"Existing data loaded from data.txt. Latest timestamp: {combined_df.index.max()}")
            print(f"Existing data loaded. Latest timestamp: {combined_df.index.max()}")
        except FileNotFoundError:
            logging.info("No existing data found. Creating new dataset.")
            print("No existing data found. Creating new dataset.")
            combined_df = pd.DataFrame()

        if not combined_df.empty:
            for interval in INTERVAL_ORDER:
                suffix = SUFFIX_MAP[interval]
                logging.info(f"Updating data for interval: {interval}")
                print(f"Updating data for interval: {interval}")
                combined_df = update_combined_data(combined_df, interval, suffix)
        else:
            logging.info("No data in data.txt, fetching data from scratch...")
            print("Fetching data from scratch...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(INTERVAL_ORDER)) as executor:
                futures = {
                    executor.submit(fetch_missing_data, symbol, interval, SUFFIX_MAP[interval], 
                                    (datetime.now(timezone.utc) - INITIAL_FETCH_DURATION), 
                                    datetime.now(timezone.utc)): interval
                    for interval in INTERVAL_ORDER
                }
                dataframes = []
                for future in concurrent.futures.as_completed(futures):
                    interval = futures[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            df = remove_duplicates(df)
                            dataframes.append(df)
                            logging.info(f"Data fetched for interval {interval}.")
                            print(f"Data fetched for interval {interval}.")
                    except Exception as exc:
                        logging.error(f'Generated an exception for interval {interval}: {exc}')
                        print(f'Generated an exception for interval {interval}: {exc}')

            if dataframes:
                combined_df = combine_interval_data(dataframes)
            else:
                combined_df = pd.DataFrame()
                logging.warning("No data was fetched from any interval.")
                print("No data was fetched from any interval.")

        if not combined_df.empty:
            logging.info(f"Saving updated data. New latest timestamp: {combined_df.index.max()}")
            print(f"Saving updated data. New latest timestamp: {combined_df.index.max()}")
            # Reset index to have 'timestamp' as a column
            combined_df.reset_index(inplace=True)

            # Define the desired column order based on INTERVAL_ORDER and SUFFIX_MAP
            desired_columns = ['timestamp']
            for interval in INTERVAL_ORDER:
                suffix = SUFFIX_MAP[interval]
                desired_columns += [
                    f'volume_{interval}',
                    f'vw{suffix}',
                    f'open_{interval}',
                    f'close_{interval}',
                    f'high_{interval}',
                    f'low_{interval}',
                    f't{suffix}',
                    f'n{suffix}'
                ]

            # Append 'vwap_<interval>' columns at the end
            for interval in INTERVAL_ORDER:
                desired_columns.append(f'vwap_{interval}')

            # Add any missing columns with NaN
            for col in desired_columns:
                if col not in combined_df.columns:
                    combined_df[col] = pd.NA

            # Reorder columns
            combined_df = combined_df[desired_columns]

            # Set 'timestamp' back as index
            combined_df.set_index('timestamp', inplace=True)

            logging.info(f"Saving updated data to data.txt")
            print(f"Saving updated data to data.txt")
            combined_df.to_csv('data.txt', sep='\t', index=True, header=True)
            logging.info("Updated data saved to data.txt")
            print("Updated data saved to data.txt")
        else:
            logging.warning("No data to save. data.txt was not updated.")
            print("No data to save. data.txt was not updated.")

    except Exception as e:
        logging.error(f"Error fetching all missing data: {e}")
        print(f"Error fetching all missing data: {e}")


def update_data_during_market_hours():
    """
    Fetches the latest data during market hours for all intervals.
    """
    if not is_market_day():
        logging.info("Today is not a market day. Skipping data update.")
        print("Today is not a market day. Skipping data update.")
        return

    logging.info("Fetching latest data during market hours...")
    print("Fetching latest data during market hours...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(INTERVAL_ORDER)) as executor:
        futures = {
            executor.submit(fetch_latest_data, symbol, interval, SUFFIX_MAP[interval]): interval
            for interval in INTERVAL_ORDER
        }
        for future in concurrent.futures.as_completed(futures):
            interval = futures[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f'Generated an exception for interval {interval}: {exc}')
                print(f'Generated an exception for interval {interval}: {exc}')


def is_market_hours():
    """
    Checks if the current time is within market hours (08:00 AM to 04:00 PM US/Eastern).
    Accounts for timezone differences.
    """
    eastern = ZoneInfo("US/Eastern")
    now_eastern = datetime.now(eastern)
    now_time = now_eastern.time()
    market_open = datetime.strptime("08:00", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    return market_open <= now_time <= market_close


def scheduled_task():
    """
    The task to run every minute during market hours.
    Logs each update attempt.
    """
    if is_market_hours():
        logging.info("Initiating scheduled data update.")
        print("Initiating scheduled data update.")
        start_update_time = datetime.now(timezone.utc)
        update_data_during_market_hours()
        end_update_time = datetime.now(timezone.utc)
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
    # Define US/Eastern timezone for scheduling
    eastern = ZoneInfo("US/Eastern")

    # Schedule fetch_all_missing_data to run at 08:00 AM US/Eastern from Monday to Friday
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
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Start the quit listener thread
    listener_thread = threading.Thread(target=quit_listener, daemon=True)
    listener_thread.start()

    # Wait for both threads to finish
    scheduler_thread.join()
    listener_thread.join()
    logging.info("Scheduler has stopped.")
    print("Scheduler has stopped.")


if __name__ == "__main__":
    # Define the date range
    # Use UTC time to avoid timezone discrepancies
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=5 * 365)  # 5 years ago

    # On program start, fetch missing data immediately
    fetch_all_missing_data()

    # Start the scheduler and quit listener
    start_scheduler()
