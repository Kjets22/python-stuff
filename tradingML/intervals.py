
import requests
import pandas as pd
from datetime import datetime

# Polygon.io API key
api_key = 'LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg'

# Symbol to fetch data for
symbol = 'SPY'

# Function to fetch all available 1-minute data from Polygon.io
def get_all_1min_data_polygon(symbol, api_key, limit=50000):
    base_url = 'https://api.polygon.io/v2/aggs/ticker'
    interval = '1'
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    all_data = []
    
    while True:
        url = f"{base_url}/{symbol}/range/{interval}/minute/2000-01-01/{end_date}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"API Request failed with status code {response.status_code}: {response.text}")
        
        data = response.json().get('results', [])
        if not data:
            break
        
        all_data.extend(data)
        
        # Get the timestamp of the earliest data point in this batch
        earliest_timestamp = data[0]['t']
        # Convert the timestamp to a date format for the next request
        end_date = datetime.utcfromtimestamp(earliest_timestamp / 1000).strftime('%Y-%m-%d')
        
        if len(data) < limit:
            break  # No more data to fetch, exit the loop
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    
    return df

# Fetch all 1-minute data
data_1min = get_all_1min_data_polygon(symbol, api_key)

# Determine the earliest timestamp
if not data_1min.empty:
    earliest_timestamp = data_1min['timestamp'].min()
    latest_timestamp = data_1min['timestamp'].max()
    print(f"Earliest 1-minute data timestamp: {earliest_timestamp}")
    print(f"Latest 1-minute data timestamp: {latest_timestamp}")
    print(f"Total duration: {latest_timestamp - earliest_timestamp}")
else:
    print("No data available for the specified symbol and interval.")
