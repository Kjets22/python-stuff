import requests
import pandas as pd
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
start_date = (datetime.today() - timedelta(days=int(4.5 * 365))).strftime('%Y-%m-%d')  # 4.5 years ago

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
                print(f"API Request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Exception occurred: {e}. Retrying...")
        time.sleep(1)  # Wait before retrying
    return pd.DataFrame()

# Function to remove duplicate indices
def remove_duplicates(df):
    if df.index.duplicated().any():
        print(f"Removing {df.index.duplicated().sum()} duplicate entries.")
        df = df[~df.index.duplicated(keep='first')]
    return df

# Function to forward-fill missing data
def forward_fill_data(df):
    # Forward-fill all missing values for each column
    for col in df.columns:
        df[col] = df[col].ffill()
    return df

# Start the timer
start_time = time.time()

# Fetch data in a loop
def fetch_all_data(symbol, interval, start_date, end_date):
    all_data = []
    current_end_date = datetime.strptime(end_date, '%Y-%m-%d')
    current_start_date = current_end_date - timedelta(days=60)  # 2 months earlier

    while current_end_date > datetime.strptime(start_date, '%Y-%m-%d'):
        current_start_str = current_start_date.strftime('%Y-%m-%d')
        current_end_str = current_end_date.strftime('%Y-%m-%d')
        print(f"Fetching data from {current_start_str} to {current_end_str}")

        df = fetch_data(symbol, interval, current_start_str, current_end_str)
        if not df.empty:
            df = remove_duplicates(df)  # Remove duplicates before appending
            all_data.append(df)
            current_end_date = current_start_date - timedelta(minutes=1)
            current_start_date = current_end_date - timedelta(days=60)
            print(f"Latest date reached: {df.index.min()}")
        else:
            break

    return pd.concat(all_data) if all_data else pd.DataFrame()

# Use threading for faster execution
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(fetch_all_data, symbol, interval, start_date, end_date): interval for interval in interval_map.keys()}
    dataframes = {}
    for future in concurrent.futures.as_completed(futures):
        try:
            interval = futures[future]
            df = future.result()
            if not df.empty:
                df = df.reset_index().drop_duplicates(subset=['timestamp']).set_index('timestamp')  # Ensure unique indices
                dataframes[interval] = df
        except Exception as exc:
            print(f'Generated an exception: {exc}')

# Combine all dataframes into a single dataframe
if dataframes:
    # Resample to minute intervals for all data
    for interval, df in dataframes.items():
        dataframes[interval] = forward_fill_data(df)  # Forward-fill missing data for each interval

    # Combine all the intervals into a single DataFrame
    data_combined = pd.concat(dataframes.values(), axis=1, join='outer')  # Using 'outer' join to ensure no loss of data
    data_combined = remove_duplicates(data_combined)  # Ensure no duplicates in the combined dataframe
    data_combined = forward_fill_data(data_combined)  # Forward-fill the combined data

    # End the timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the latest date in the combined dataset
    latest_date = data_combined.index.max()

    print(data_combined.head())
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Latest date reached: {latest_date}")

    # Save the combined data to a text file
    data_combined.to_csv('combined_data.txt', sep='\t', index=True, header=True)
    print("Data saved to combined_data.txt")
else:
    print("No data was fetched.")
