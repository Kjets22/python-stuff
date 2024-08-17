import requests
import pandas as pd

api_key = 'OmJlOGNkM2FiZjAyYTk0OGZmOThmOWU0M2I0Yzg2ODgy'
symbol = 'AAPL'

def fetch_iex_data(symbol, interval):
    url = f'https://cloud.iexapis.com/stable/stock/{symbol}/chart/{interval}?token={api_key}'
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

# Fetch data
data_15min = fetch_iex_data(symbol, '5dm')  # 5 days minute data
data_1hour = fetch_iex_data(symbol, '1d')   # 1 day data for hourly

# Save data to CSV files
data_15min.to_csv('AAPL_15min.csv')
data_1hour.to_csv('AAPL_1hour.csv')

print("Data saved to files: 'AAPL_15min.csv', 'AAPL_1hour.csv'")
