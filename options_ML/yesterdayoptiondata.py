import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to get options data for a specific date
def get_options_data_frame(api_key, symbol, date):
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

    def find_closest_options(options_df, current_price):
        if options_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
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

    def get_previous_close_price(symbol, date, api_key):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}&outputsize=full'
        response = requests.get(url)
        data = response.json()

        if 'Time Series (1min)' in data:
            ts = data['Time Series (1min)']
            df = pd.DataFrame.from_dict(ts, orient='index')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            trading_day = df.index[df.index.date == date]
            if not trading_day.empty:
                close_price = df.loc[trading_day[-1], '4. close']
                return float(close_price)
        
        raise ValueError(f"No stock data found for {symbol} on {date}.")

    def get_last_trading_day(date):
        while date.weekday() > 4:
            date -= timedelta(days=1)
        return date

    try:
        specified_date = datetime.strptime(date, '%Y-%m-%d')
        yesterday = get_last_trading_day(specified_date - timedelta(days=1))
        day_before_yesterday = get_last_trading_day(yesterday - timedelta(days=1))

        previous_close_price = get_previous_close_price(symbol, yesterday.date(), api_key)

        options_yesterday = get_options_data(symbol, yesterday.strftime('%Y-%m-%d'), api_key)
        options_day_before_yesterday = get_options_data(symbol, day_before_yesterday.strftime('%Y-%m-%d'), api_key)

        if options_yesterday.empty or options_day_before_yesterday.empty:
            raise ValueError("Insufficient options data available for the given dates.")

        min_expiration_yesterday = options_yesterday['expiration'].min()
        min_expiration_day_before_yesterday = options_day_before_yesterday['expiration'].min()
        later_expiration = max(min_expiration_yesterday, min_expiration_day_before_yesterday)

        options_yesterday = options_yesterday[options_yesterday['expiration'] == later_expiration]
        options_day_before_yesterday = options_day_before_yesterday[options_day_before_yesterday['expiration'] == later_expiration]

        calls_yesterday, puts_yesterday = find_closest_options(options_yesterday, previous_close_price)
        calls_day_before_yesterday, puts_day_before_yesterday = find_closest_options(options_day_before_yesterday, previous_close_price)

        calls_yesterday['Open Price'] = pd.to_numeric(calls_day_before_yesterday['Close Price'].values)
        puts_yesterday['Open Price'] = pd.to_numeric(puts_day_before_yesterday['Close Price'].values)

        calls_yesterday['Close Price'] = pd.to_numeric(calls_yesterday['Close Price'])
        puts_yesterday['Close Price'] = pd.to_numeric(puts_yesterday['Close Price'])

        calls_yesterday['Change'] = calls_yesterday['Close Price'] - calls_yesterday['Open Price']
        puts_yesterday['Change'] = puts_yesterday['Close Price'] - puts_yesterday['Open Price']

        calls_yesterday = calls_yesterday[['Strike Price', 'Type', 'Open Price', 'Close Price', 'Change', 'Volume', 'Expiration Date']]
        puts_yesterday = puts_yesterday[['Strike Price', 'Type', 'Open Price', 'Close Price', 'Change', 'Volume', 'Expiration Date']]

        combined_df = pd.concat([calls_yesterday, puts_yesterday])

        combined_df['Date'] = yesterday

        return combined_df

    except ValueError as e:
        print(e)
        return pd.DataFrame()

# Main execution
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

# Define today's date
today = datetime.now().strftime('%Y-%m-%d')
date="2024-08-07"
correctdate=datetime.strptime(date,'%Y-%m-%d')
# Get today's options data
options_df = get_options_data_frame(api_key, symbol, today)

# Append the data to the existing CSV file
if not options_df.empty:
    options_df.to_csv('options_data.csv', mode='a', header=False, index=False)
    print(f"Today's options data appended to 'options_data.csv'.")
else:
    print("No options data to append.")
