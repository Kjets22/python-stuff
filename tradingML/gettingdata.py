import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import concurrent.futures

# Polygon.io API key
api_key = 'LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg'

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
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')  # 5 years ago

# Function to fetch data from Polygon.io
def fetch_data(symbol, interval, start_date, end_date, retries=3):
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    url = f"{base_url}/{symbol}/range/{interval_map[interval]}/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    
    for _ in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
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
                        'v': f'volume_{interval}'
                    }, inplace=True)
                    return df
            else:
                print(f"API Request failed for {interval} interval with status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Exception occurred for {interval} interval: {e}. Retrying...")
        time.sleep(1)  # Wait before retrying
    return pd.DataFrame()

# Function to remove duplicate indices and ensure unique index values
def remove_duplicates(df):
    if df.index.duplicated().any():
        print(f"Removing {df.index.duplicated().sum()} duplicate entries.")
        df = df[~df.index.duplicated(keep='first')]
    return df

# Function to forward-fill missing data
def forward_fill_data(df):
    for col in df.columns:
        df[col] = df[col].ffill()
    return df

# Fetch missing data and ensure no duplicates
def fetch_missing_data(symbol, interval, start_date, end_date):
    all_data = []
    current_end_date = datetime.strptime(end_date, '%Y-%m-%d')
    current_start_date = current_end_date - timedelta(days=60)  # 2 months earlier

    while current_end_date > datetime.strptime(start_date, '%Y-%m-%d'):
        current_start_str = current_start_date.strftime('%Y-%m-%d')
        current_end_str = current_end_date.strftime('%Y-%m-%d')
        print(f"Fetching {interval} data from {current_start_str} to {current_end_str}")

        df = fetch_data(symbol, interval, current_start_str, current_end_str)
        if not df.empty:
            df = remove_duplicates(df)  # Remove duplicates right after fetching
            all_data.append(df)
            current_end_date = current_start_date - timedelta(minutes=1)
            current_start_date = current_end_date - timedelta(days=60)
            print(f"Latest date reached for {interval}: {df.index.min()}")
        else:
            break

    return pd.concat(all_data) if all_data else pd.DataFrame()

# Function to update the combined data if missing data is found
def update_combined_data(combined_df, interval):
    print(f"Checking for missing data for interval: {interval}...")

    # Get the latest date in the existing combined data
    latest_date_combined = combined_df.index.max()
    current_end_date = datetime.today()

    if latest_date_combined < current_end_date:
        print(f"Data is missing from {latest_date_combined} to {current_end_date} for {interval}. Fetching missing data...")

        new_data_df = fetch_missing_data(symbol, interval, latest_date_combined.strftime('%Y-%m-%d'), current_end_date.strftime('%Y-%m-%d'))
        if not new_data_df.empty:
            new_data_df = remove_duplicates(new_data_df)  # Ensure no duplicates after fetching new data
            combined_df = pd.concat([combined_df, new_data_df], axis=1)
            combined_df = remove_duplicates(combined_df)  # Remove duplicates after concatenating
            combined_df = combined_df.resample('min').ffill()  # Forward fill any missing data

    return combined_df

# Function to combine the interval data without duplicates
def combine_interval_data(dataframes):
    print("Combining data from all intervals...")
    combined_df = pd.concat(dataframes, axis=1)  # Concatenate along columns
    combined_df = remove_duplicates(combined_df)  # Ensure no duplicate index values
    combined_df = forward_fill_data(combined_df)
    return combined_df

# Load existing testing.txt instead of combined_data.txt
try:
    combined_df = pd.read_csv('testing.txt', sep='\t', index_col='timestamp', parse_dates=True)
    print("Existing data loaded from testing.txt.")
except FileNotFoundError:
    print("No existing data found. Creating new dataset.")
    combined_df = pd.DataFrame()

# Start the timer
start_time = time.time()

# Check and update for missing data
if not combined_df.empty:
    for interval in interval_map.keys():
        combined_df = update_combined_data(combined_df, interval)
else:
    print("No data in testing.txt, fetching data from scratch...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_missing_data, symbol, interval, start_date, end_date): interval for interval in interval_map.keys()}
        dataframes = []
        for future in concurrent.futures.as_completed(futures):
            try:
                df = future.result()
                if not df.empty:
                    df = remove_duplicates(df)  # Remove duplicates right after fetching each interval
                    dataframes.append(df)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
    
    if dataframes:
        combined_df = combine_interval_data(dataframes)

# End the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Print the latest date in the combined dataset
latest_date = combined_df.index.max()

print(combined_df.head())
print(f"Time taken: {elapsed_time:.2f} seconds")
print(f"Latest date reached: {latest_date}")

# Save the updated combined data to testing.txt instead of combined_data.txt
combined_df.to_csv('testing.txt', sep='\t', index=True, header=True)
print("Updated data saved to testing.txt")
