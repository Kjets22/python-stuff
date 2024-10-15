
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

# Polygon.io API key
api_key = 'LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg'

# Symbol to fetch data for
symbol = 'SPY'

# Interval to fetch (you can change this to 5min, 15min, etc.)
interval = '30min'

# Define the date range
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')  # Last 5 days for testing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Function to fetch data from Polygon.io
def fetch_data(symbol, interval, start_date, end_date, retries=3):
    interval_map = {
        '1min': '1',
        '5min': '5',
        '15min': '15',
        '30min': '30',
        '60min': '60'
    }
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    url = f"{base_url}/{symbol}/range/{interval_map[interval]}/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    
    logging.info(f"Fetching data from URL: {url}")
    
    for _ in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json().get('results', [])
                logging.info(f"Data fetched for {interval}: {len(data)} rows.")
                if data:
                    # Convert the data into a DataFrame
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
                    logging.warning(f"No data returned for {interval}.")
            else:
                logging.error(f"API Request failed for {interval} interval with status code {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"Exception occurred for {interval} interval: {e}. Retrying...")
        # Wait before retrying
        logging.info("Retrying after 1 second...")
        time.sleep(1)
    return pd.DataFrame()

# Fetch and print the data
if __name__ == "__main__":
    df = fetch_data(symbol, interval, start_date, end_date)
    if not df.empty:
        print(f"Data fetched for {interval} interval:")
        print(df.head())  # Print first 5 rows for verification
    else:
        print(f"No data fetched for {interval} interval.")
