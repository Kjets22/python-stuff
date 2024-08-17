import requests
import pandas as pd
from datetime import datetime, timedelta

# Define the API key and symbol
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

# Function to get options data for a specific date
def get_options_data(symbol, date, api_key):
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&date={date}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if 'data' in data:
        options_df = pd.DataFrame(data['data'])
        if 'strike' in options_df.columns and 'expiration' in options_df.columns:
            options_df['strike'] = pd.to_numeric(options_df['strike'], errors='coerce')
            options_df['expiration'] = pd.to_datetime(options_df['expiration'], errors='coerce')
            return options_df
        else:
            print(f"Missing necessary columns in options data for {symbol} on {date}.")
    else:
        print(f"No options data found for {symbol} on {date}.")
    
    return pd.DataFrame()

# Function to find closest options with the soonest expiration date
def find_closest_options(options_df, current_price):
    if options_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter options with the soonest expiration date
    min_expiration = options_df['expiration'].min()
    options_df = options_df[options_df['expiration'] == min_expiration]
    
    options_df['distance_to_price'] = abs(options_df['strike'] - current_price)
    
    calls_above = options_df[(options_df['type'] == 'call') & (options_df['strike'] >= current_price)].sort_values(by='strike').head(2)
    calls_below = options_df[(options_df['type'] == 'call') & (options_df['strike'] < current_price)].sort_values(by='strike', ascending=False).head(2)
    closest_calls = pd.concat([calls_above, calls_below])
    
    puts_above = options_df[(options_df['type'] == 'put') & (options_df['strike'] >= current_price)].sort_values(by='strike').head(2)
    puts_below = options_df[(options_df['type'] == 'put') & (options_df['strike'] < current_price)].sort_values(by='strike', ascending=False).head(2)
    closest_puts = pd.concat([puts_above, puts_below])
    
    closest_calls = closest_calls[['strike', 'type', 'last', 'volume', 'expiration']]
    closest_puts = closest_puts[['strike', 'type', 'last', 'volume', 'expiration']]
    
    closest_calls.columns = ['Strike Price', 'Type', 'Close Price', 'Volume', 'Expiration Date']
    closest_puts.columns = ['Strike Price', 'Type', 'Close Price', 'Volume', 'Expiration Date']
    
    return closest_calls, closest_puts

# Function to get the stock's close price for a specific date
def get_previous_close_price(symbol, date, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()

    if 'Time Series (1min)' in data:
        ts = data['Time Series (1min)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Get the trading day
        trading_day = df.index[df.index.date == date]
        if not trading_day.empty:
            close_price = df.loc[trading_day[-1], '4. close']
            return float(close_price)
    
    raise ValueError(f"No stock data found for {symbol} on {date}.")

# Determine the dates
today = datetime.now()
yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
day_before_yesterday = (today - timedelta(days=2)).strftime('%Y-%m-%d')

# Ensure we handle weekends and holidays
def get_last_trading_day(date):
    while date.weekday() > 4:  # 0 = Monday, 4 = Friday
        date -= timedelta(days=1)
    return date

try:
    # Get the last trading days
    yesterday = get_last_trading_day(today - timedelta(days=1))
    day_before_yesterday = get_last_trading_day(yesterday - timedelta(days=1))

    # Get the stock's previous close price
    previous_close_price = get_previous_close_price(symbol, yesterday.date(), api_key)
    
    # Get options data for yesterday and the previous stock market open day
    options_yesterday = get_options_data(symbol, yesterday.strftime('%Y-%m-%d'), api_key)
    options_day_before_yesterday = get_options_data(symbol, day_before_yesterday.strftime('%Y-%m-%d'), api_key)
    
    # Ensure options data is available
    if options_yesterday.empty or options_day_before_yesterday.empty:
        raise ValueError("Insufficient options data available for the given dates.")
    
    # Find the later expiration date between the two days
    min_expiration_yesterday = options_yesterday['expiration'].min()
    min_expiration_day_before_yesterday = options_day_before_yesterday['expiration'].min()
    later_expiration = max(min_expiration_yesterday, min_expiration_day_before_yesterday)

    # Filter options to only include those with the later expiration date
    options_yesterday = options_yesterday[options_yesterday['expiration'] == later_expiration]
    options_day_before_yesterday = options_day_before_yesterday[options_day_before_yesterday['expiration'] == later_expiration]

    # Find closest options for yesterday based on previous day's close price
    calls_yesterday, puts_yesterday = find_closest_options(options_yesterday, previous_close_price)
    
    # Find closest options for the day before yesterday based on the previous day's close price
    calls_day_before_yesterday, puts_day_before_yesterday = find_closest_options(options_day_before_yesterday, previous_close_price)
    
    # Print yesterday's options
    print(f"Options data for {symbol} on {yesterday.date()} (based on the previous close price ${previous_close_price:.2f}):")
    print("\nCalls:")
    print(calls_yesterday.to_string(index=False))
    print("\nPuts:")
    print(puts_yesterday.to_string(index=False))
    
    # Print the day before yesterday's options
    print(f"\nOptions data for {symbol} on {day_before_yesterday.date()} (based on the close price from two days ago ${previous_close_price:.2f}):")
    print("\nCalls:")
    print(calls_day_before_yesterday.to_string(index=False))
    print("\nPuts:")
    print(puts_day_before_yesterday.to_string(index=False))

except ValueError as e:
    print(e)
