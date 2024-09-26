import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import concurrent.futures
import logging
import os
import schedule

# ---------------------------- Configuration ----------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Polygon.io API key
api_key = 'LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg'  # Replace with your actual Polygon.io API key

# Symbol to fetch data for
symbol = 'SPY'

# Define the intervals and date range
interval_map = {
    '1min': '1',
    '5min': '5',
    '15min': '15',
    '30min': '30',
    '60min': '60'
}

# Output file
DATA_FILE = 'data.txt'

# Time retention settings
DATA_RETENTION_MINUTES = 1000  # Number of latest minutes to retain in data.txt

# Fetching settings
FETCH_LIMIT = 50000  # Maximum number of records per API call
RETRIES = 3  # Number of retries for API requests
SLEEP_BETWEEN_RETRIES = 1  # Seconds to wait before retrying after a failed attempt

# ---------------------------- Helper Functions ----------------------------

def fetch_data(symbol, interval, start_date, end_date, retries=RETRIES):
    """
    Fetches data from Polygon.io for a given symbol, interval, and date range.
    
    Parameters:
    - symbol (str): Stock symbol (e.g., 'SPY').
    - interval (str): Time interval (e.g., '1min').
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - retries (int): Number of retry attempts for failed API calls.
    
    Returns:
    - pd.DataFrame: DataFrame containing fetched data. Empty if fetch fails.
    """
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    url = f"{base_url}/{symbol}/range/{interval_map[interval]}/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit={FETCH_LIMIT}&apiKey={api_key}"
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises HTTPError for bad responses
            data = response.json().get('results', [])
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.rename(columns={
                    'o': f'open_{interval}',
                    'h': f'high_{interval}',
                    'l': f'low_{interval}',
                    'c': f'close_{interval}',
                    'v': f'volume_{interval}',
                    'n': f'transactions_{interval}'
                }, inplace=True)
                logging.info(f"Fetched {len(df)} records for interval '{interval}' from {start_date} to {end_date}.")
                return df
            else:
                logging.warning(f"No data returned for interval '{interval}' from {start_date} to {end_date}.")
                return pd.DataFrame()
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}. Attempt {attempt} of {retries}.")
        except Exception as err:
            logging.error(f"An error occurred: {err}. Attempt {attempt} of {retries}.")
        time.sleep(SLEEP_BETWEEN_RETRIES)
    logging.error(f"Failed to fetch data for interval '{interval}' from {start_date} to {end_date} after {retries} attempts.")
    return pd.DataFrame()

def remove_duplicates(df):
    """
    Removes duplicate indices from the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    
    Returns:
    - pd.DataFrame: DataFrame without duplicate indices.
    """
    if df.index.duplicated().any():
        num_duplicates = df.index.duplicated().sum()
        logging.warning(f"Removing {num_duplicates} duplicate entries.")
        df = df[~df.index.duplicated(keep='first')]
    return df

def forward_fill_data(df):
    """
    Forward-fills missing data in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    
    Returns:
    - pd.DataFrame: DataFrame with forward-filled data.
    """
    df = df.ffill()
    df = df.fillna(0)  # Replace any remaining NaNs with 0
    return df

def fetch_missing_data(symbol, interval, start_date, end_date):
    """
    Fetches missing data for a specific interval.
    
    Parameters:
    - symbol (str): Stock symbol.
    - interval (str): Time interval.
    - start_date (str): Start date.
    - end_date (str): End date.
    
    Returns:
    - pd.DataFrame: DataFrame containing fetched data.
    """
    logging.info(f"Fetching missing data for interval '{interval}' from {start_date} to {end_date}.")
    df = fetch_data(symbol, interval, start_date, end_date)
    if not df.empty:
        df = remove_duplicates(df)
    return df

def update_data_file(symbol, interval_map, data_file, data_retention_minutes):
    """
    Updates the data file with the latest fetched data.
    
    Parameters:
    - symbol (str): Stock symbol.
    - interval_map (dict): Mapping of interval labels to numeric values.
    - data_file (str): Path to the data file.
    - data_retention_minutes (int): Number of latest minutes to retain.
    """
    # Load existing data
    if os.path.exists(data_file):
        try:
            combined_df = pd.read_csv(data_file, sep='\t', index_col='timestamp', parse_dates=True)
            logging.info(f"Loaded existing data from '{data_file}'.")
        except Exception as e:
            logging.error(f"Error reading '{data_file}': {e}. Starting with empty DataFrame.")
            combined_df = pd.DataFrame()
    else:
        logging.info(f"No existing data found at '{data_file}'. Creating new dataset.")
        combined_df = pd.DataFrame()
    
    # Determine the latest timestamp in the existing data
    if not combined_df.empty:
        latest_timestamp = combined_df.index.max()
    else:
        latest_timestamp = None
    
    # Define the time window for fetching new data
    now = datetime.utcnow()
    delay_minutes = 5  # Adjust if necessary to account for data latency
    fetch_end_time = now - timedelta(minutes=delay_minutes)
    
    for interval in interval_map.keys():
        # Determine the start and end dates for fetching
        if latest_timestamp:
            # Fetch data after the latest timestamp
            fetch_start_time = latest_timestamp + timedelta(minutes=1)
            if fetch_start_time >= fetch_end_time:
                logging.info(f"No new data to fetch for interval '{interval}'.")
                continue
            start_date_str = fetch_start_time.strftime('%Y-%m-%d')
        else:
            # No existing data, fetch historical data from the start_date
            # Adjust the start_date as needed
            start_date = now - timedelta(days=60)  # Fetch last 60 days
            start_date_str = start_date.strftime('%Y-%m-%d')
        
        end_date_str = fetch_end_time.strftime('%Y-%m-%d')
        
        # Fetch missing data
        df_new = fetch_missing_data(symbol, interval, start_date_str, end_date_str)
        if not df_new.empty:
            # Append new data to the combined DataFrame
            combined_df = pd.concat([combined_df, df_new])
            combined_df = remove_duplicates(combined_df)
            logging.info(f"Appended {len(df_new)} new records for interval '{interval}'.")
        else:
            logging.info(f"No new data fetched for interval '{interval}'.")
    
    # Forward-fill missing data
    combined_df = forward_fill_data(combined_df)
    
    # Retain only the latest 'data_retention_minutes' of data
    if not combined_df.empty:
        combined_df = combined_df.sort_index().tail(data_retention_minutes)
    
    # Save the updated data to the file
    try:
        combined_df.to_csv(data_file, sep='\t', index=True, header=True)
        logging.info(f"Data saved to '{data_file}'. Total records: {len(combined_df)}.")
    except Exception as e:
        logging.error(f"Error saving data to '{data_file}': {e}.")

def run_extractingdata(symbol, interval_map, data_file, data_retention_minutes):
    """
    Runs the data extraction process.
    
    Parameters:
    - symbol (str): Stock symbol.
    - interval_map (dict): Mapping of interval labels to numeric values.
    - data_file (str): Path to the data file.
    - data_retention_minutes (int): Number of latest minutes to retain.
    """
    update_data_file(symbol, interval_map, data_file, data_retention_minutes)

def main():
    """
    Main function to run the data extraction process.
    """
    run_extractingdata(symbol, interval_map, DATA_FILE, DATA_RETENTION_MINUTES)

# ---------------------------- Scheduling ----------------------------

def schedule_data_fetch(symbol, interval_map, data_file, data_retention_minutes, interval_minutes=1):
    """
    Schedules the data fetching to run at specified intervals.
    
    Parameters:
    - symbol (str): Stock symbol.
    - interval_map (dict): Mapping of interval labels to numeric values.
    - data_file (str): Path to the data file.
    - data_retention_minutes (int): Number of latest minutes to retain.
    - interval_minutes (int): Frequency in minutes to fetch data.
    """
    schedule.every(interval_minutes).minutes.do(run_extractingdata, symbol, interval_map, data_file, data_retention_minutes)
    logging.info(f"Scheduled data fetching every {interval_minutes} minute(s).")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# ---------------------------- Execution ----------------------------

if __name__ == "__main__":
    try:
        # Option 1: Run once and exit
        # main()
        
        # Option 2: Schedule to run at regular intervals (e.g., every minute)
        schedule_data_fetch(symbol, interval_map, DATA_FILE, DATA_RETENTION_MINUTES, interval_minutes=1)
    except KeyboardInterrupt:
        logging.info("Data extraction terminated by user.")
