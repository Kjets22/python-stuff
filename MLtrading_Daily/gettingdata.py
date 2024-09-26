import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import concurrent.futures
import os
import logging
import apikey

# Import technical analysis indicators
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# ---------------------------- Configuration ----------------------------

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("gettingdata.log"),
        logging.StreamHandler()
    ]
)

# Define global constants
symbol = 'SPY'  # Stock symbol to fetch data for
intervals = ['1min', '5min', '15min', '30min', '60min']  # List of intervals
api_key = apikey.API_KEY # Replace with your actual Polygon.io API key

if not api_key:
    logging.error("API key not found. Please set the 'api_key' variable in the script.")
    raise ValueError("API key not found. Please set the 'api_key' variable in the script.")

interval_map = {
    '1min': '1',
    '5min': '5',
    '15min': '15',
    '30min': '30',
    '60min': '60'
}

# Define the maximum look-back period for fetching historical data
MAX_LOOKBACK_DAYS = 4.5 * 365  # 4.5 years

# ---------------------------- Helper Functions ----------------------------

def fetch_data(symbol, interval, start_date, end_date, retries=3, rate_limit_delay=1):
    """
    Fetches data from Polygon.io for a given symbol and interval between start_date and end_date.
    """
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    url = f"{base_url}/{symbol}/range/{interval_map[interval]}/minute/{int(start_date.timestamp() * 1000)}/{int(end_date.timestamp() * 1000)}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json().get('results', [])
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')  # Timezone-naive
                    df.set_index('timestamp', inplace=True)
                    df.rename(columns={
                        'o': f'open_{interval}',
                        'h': f'high_{interval}',
                        'l': f'low_{interval}',
                        'c': f'close_{interval}',
                        'v': f'volume_{interval}'
                    }, inplace=True)
                    logging.info(f"Successfully fetched data for interval {interval}.")
                    return df
                else:
                    logging.info(f"No new data found for interval {interval} from {start_date} to {end_date}.")
                    return pd.DataFrame()
            elif response.status_code == 403:
                # API access denied for this interval
                error_message = response.json().get('message', 'No message provided.')
                logging.warning(f"API Request failed for interval {interval} with status code 403: {error_message}")
                return None  # Indicate failure due to API limitations
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))  # Default to 60 seconds
                logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                continue  # Retry the request
            else:
                logging.warning(f"API Request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"Exception occurred during data fetch: {e}. Attempt {attempt + 1} of {retries}.")
        time.sleep(rate_limit_delay)  # Wait before retrying
    logging.error(f"Failed to fetch data for interval {interval} after {retries} attempts.")
    return pd.DataFrame()

def remove_duplicates(df):
    """
    Removes duplicate entries based on the timestamp index.
    """
    if df.index.duplicated().any():
        num_duplicates = df.index.duplicated().sum()
        logging.info(f"Removing {num_duplicates} duplicate entries.")
        df = df[~df.index.duplicated(keep='first')]
    return df

def forward_fill_data(df):
    """
    Forward-fills all missing values for each column in the DataFrame.
    """
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_last_timestamp(combined_df):
    """
    Retrieves the last timestamp from the combined dataframe.
    """
    if combined_df.empty:
        # If the DataFrame is empty, set a default start date (e.g., 4.5 years ago)
        return datetime.utcnow() - timedelta(days=MAX_LOOKBACK_DAYS)
    return combined_df.index.max()

def fetch_new_data(symbol, interval, last_timestamp, end_date, retries=3, rate_limit_delay=1):
    """
    Fetches new data from the last_timestamp to end_date for the given symbol and interval.
    """
    df = fetch_data(symbol, interval, last_timestamp, end_date, retries, rate_limit_delay)
    return df

def fill_missing_data(combined_df, interval, new_start_time, end_date):
    """
    Fills missing data for a specific interval by carrying forward the last available data.
    
    Parameters:
    - combined_df (pd.DataFrame): The existing combined dataframe.
    - interval (str): The interval being processed.
    - new_start_time (datetime): The start time for filling data.
    - end_date (datetime): The end time up to which data is needed.
    
    Returns:
    - pd.DataFrame: Updated combined dataframe with filled data for the interval.
    """
    # Generate a date range from new_start_time to end_date with 1-minute frequency
    date_range = pd.date_range(start=new_start_time, end=end_date, freq='T')
    
    # Get the last known data point for the interval
    last_known_time = combined_df.index.max()
    if last_known_time in combined_df.index:
        last_known_value = combined_df.iloc[-1][f'close_{interval}']
    else:
        last_known_value = 0  # Default value if no previous data
    
    # Create a DataFrame with the date_range and the last known value
    filled_data = pd.DataFrame({
        f'open_{interval}': last_known_value,
        f'high_{interval}': last_known_value,
        f'low_{interval}': last_known_value,
        f'close_{interval}': last_known_value,
        f'volume_{interval}': 0
    }, index=date_range)
    
    # Append the filled data to combined_df
    combined_df = pd.concat([combined_df, filled_data])
    logging.info(f"Filled missing data for interval {interval} from {new_start_time} to {end_date}.")
    
    return combined_df

def update_combined_data(combined_df, interval):
    """
    Updates the combined dataframe with new data for a specific interval.
    If data fetching fails due to API limitations, fills the interval's data with the last known value.
    """
    logging.info(f"Checking for new data for interval: {interval}...")
    
    # Get the latest timestamp in the combined dataframe
    last_timestamp = get_last_timestamp(combined_df)
    
    # Define the end date for fetching data
    current_end_date = datetime.utcnow()  # Use current UTC time, naive
    
    # Define the start date for new data (last_timestamp + 1 minute)
    start_date_new = last_timestamp + timedelta(minutes=1)
    
    # If the start_date_new is in the future, skip fetching
    if start_date_new > current_end_date:
        logging.info(f"No new data to fetch for interval {interval}. Latest timestamp: {last_timestamp}")
        return combined_df
    
    # Fetch new data
    new_data_df = fetch_new_data(symbol, interval, start_date_new, current_end_date, retries=3, rate_limit_delay=1)
    
    if new_data_df is not None and not new_data_df.empty:
        # Successfully fetched new data
        new_data_df = remove_duplicates(new_data_df)
        combined_df = pd.concat([combined_df, new_data_df])
        combined_df = remove_duplicates(combined_df)
        logging.info(f"Appended {len(new_data_df)} new data points for interval {interval}.")
    else:
        # Failed to fetch data due to API limitations or no new data
        if new_data_df is None:
            # API limitation encountered
            logging.warning(f"Cannot fetch data for interval {interval} due to API limitations. Filling with last known data.")
            combined_df = fill_missing_data(combined_df, interval, start_date_new, current_end_date)
        else:
            # No new data fetched
            logging.info(f"No new data fetched for interval {interval}.")
    
    return combined_df

def fetch_missing_data(symbol, intervals, start_date, end_date):
    """
    Fetches missing data for all specified intervals.
    """
    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(intervals)) as executor:
        # Submit fetch tasks for each interval
        futures = {
            executor.submit(fetch_new_data, symbol, interval, start_date, end_date): interval for interval in intervals
        }
        for future in concurrent.futures.as_completed(futures):
            interval = futures[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    df = remove_duplicates(df)
                    all_data.append(df)
                    logging.info(f"Fetched {len(df)} new data points for interval {interval}.")
                elif df is None:
                    # API limitation encountered
                    logging.warning(f"Cannot fetch data for interval {interval} due to API limitations.")
                else:
                    logging.info(f"No new data for interval {interval}.")
            except Exception as exc:
                logging.error(f"Generated an exception for interval {interval}: {exc}")
    if all_data:
        combined_new_data = pd.concat(all_data)
        combined_new_data = remove_duplicates(combined_new_data)
        combined_new_data = forward_fill_data(combined_new_data)
        return combined_new_data
    else:
        return pd.DataFrame()

# ---------------------------- Main Function ----------------------------

def main():
    data_file = 'combined_data.txt'  # File to store combined data
    
    # Load existing combined_data.txt
    try:
        combined_df = pd.read_csv(data_file, sep='\t', index_col='timestamp', parse_dates=True)
        logging.info(f"Loaded existing data from '{data_file}'.")
    except FileNotFoundError:
        logging.info(f"No existing data found. Creating new dataset for symbol {symbol}.")
        combined_df = pd.DataFrame()
    
    # Update combined_df with new data for each interval
    if combined_df.empty:
        # If no data exists, fetch historical data for all intervals
        logging.info(f"Fetching historical data for symbol {symbol} from scratch.")
        historical_end_date = datetime.utcnow()
        historical_start_date = historical_end_date - timedelta(days=MAX_LOOKBACK_DAYS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(intervals)) as executor:
            futures = {
                executor.submit(fetch_new_data, symbol, interval, historical_start_date, historical_end_date): interval for interval in intervals
            }
            for future in concurrent.futures.as_completed(futures):
                interval = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        df = remove_duplicates(df)
                        combined_df = pd.concat([combined_df, df])
                        logging.info(f"Fetched {len(df)} data points for interval {interval}.")
                    elif df is None:
                        # API limitation encountered
                        logging.warning(f"Cannot fetch data for interval {interval} due to API limitations.")
                        # Fill missing data with last known value (which doesn't exist yet, so use default)
                        default_value = 0
                        filled_data = pd.DataFrame({
                            f'open_{interval}': default_value,
                            f'high_{interval}': default_value,
                            f'low_{interval}': default_value,
                            f'close_{interval}': default_value,
                            f'volume_{interval}': 0
                        }, index=pd.date_range(start=historical_start_date, end=historical_end_date, freq='T'))
                        combined_df = pd.concat([combined_df, filled_data])
                        logging.info(f"Filled missing data for interval {interval} with default values.")
                    else:
                        logging.info(f"No data fetched for interval {interval}.")
                except Exception as exc:
                    logging.error(f"Exception occurred while fetching data for interval {interval}: {exc}")
        # Forward-fill missing data
        combined_df = forward_fill_data(combined_df)
    else:
        # If data exists, fetch only new data since the last timestamp
        for interval in intervals:
            combined_df = update_combined_data(combined_df, interval)
    
    # Resample the combined dataframe to 1-minute intervals and forward fill
    combined_df = combined_df.resample('T').ffill()
    
    # Save the updated combined data to the text file
    combined_df.to_csv(data_file, sep='\t', index=True, header=True)
    logging.info(f"Updated data saved to '{data_file}'. Total data points: {len(combined_df)}")
    
    # Optionally, display the latest few entries
    logging.info("Latest data points:")
    logging.info(combined_df.tail())

if __name__ == "__main__":
    main()
