import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import time
import io

def get_intraday_data(symbol, interval, api_key, outputsize='full'):
    """
    Fetches intraday stock data for a given symbol and interval from Alpha Vantage API.

    Parameters:
        symbol (str): Stock symbol.
        interval (str): Time interval between data points (e.g., '1min', '5min').
        api_key (str): Alpha Vantage API key.
        outputsize (str): 'compact' or 'full'.

    Returns:
        pd.DataFrame: DataFrame containing stock data with datetime index.
    """
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'apikey': api_key,
        'outputsize': outputsize,
        'datatype': 'csv'
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API Request failed with status code {response.status_code}")
    
    data = pd.read_csv(io.StringIO(response.text))
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    data = data.sort_index()
    data = data.add_prefix(f'{interval}_')
    return data

def add_technical_indicators(df):
    """
    Adds various technical indicators to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data.

    Returns:
        pd.DataFrame: DataFrame enriched with technical indicators.
    """
    # Ensure no duplicate indices
    df = df[~df.index.duplicated(keep='first')]

    # Close price for calculations
    close = df['1min_close']

    # Moving Averages
    df['MA_20'] = close.rolling(window=20).mean()
    df['MA_50'] = close.rolling(window=50).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()

    # RSI
    rsi = RSIIndicator(close=close, window=14)
    df['RSI_14'] = rsi.rsi()

    # Bollinger Bands
    bollinger = BollingerBands(close=close, window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()

    # MACD
    macd = MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # Stochastic Oscillator
    high = df['1min_high']
    low = df['1min_low']
    stochastic = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df['Stochastic_K'] = stochastic.stoch()
    df['Stochastic_D'] = stochastic.stoch_signal()

    # Price Changes
    df['Price_Change'] = close.diff()
    df['Price_Change_Percent'] = close.pct_change() * 100

    return df

def identify_breakouts(df, price_change_threshold=1.0, rsi_threshold=70, macd_threshold=0):
    """
    Identifies different types of breakout events based on specified criteria.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data with technical indicators.
        price_change_threshold (float): Minimum price change to qualify as a breakout.
        rsi_threshold (float): RSI value to consider for overbought/oversold conditions.
        macd_threshold (float): MACD value to consider for bullish/bearish momentum.

    Returns:
        pd.DataFrame: DataFrame with additional columns indicating breakout events.
    """
    # Calculate volume surge
    df['Volume_Surge'] = df['1min_volume'] > df['1min_volume'].rolling(window=20).mean() * 1.5

    df['Upward_Breakout'] = (
        (df['Price_Change'] >= price_change_threshold) &
        (df['RSI_14'] > rsi_threshold) &
        (df['MACD'] > macd_threshold) &
        (df['1min_close'] > df['Bollinger_High'])
    )

    df['Downward_Breakout'] = (
        (df['Price_Change'] <= -price_change_threshold) &
        (df['RSI_14'] < (100 - rsi_threshold)) &
        (df['MACD'] < -macd_threshold) &
        (df['1min_close'] < df['Bollinger_Low'])
    )

    df['Reversal_Breakout'] = (
        (df['Price_Change'].shift(1) < 0) &
        (df['Price_Change'] > price_change_threshold) &
        (df['MACD'].diff() > 0)
    )

    df['Resistance_Breakout'] = (
        (df['1min_close'] > df['1min_close'].rolling(window=50).max().shift(1)) &
        (df['Volume_Surge'])
    )

    df['Support_Breakout'] = (
        (df['1min_close'] < df['1min_close'].rolling(window=50).min().shift(1)) &
        (df['Volume_Surge'])
    )

    # Combine all breakouts
    df['Any_Breakout'] = df[['Upward_Breakout', 'Downward_Breakout', 'Reversal_Breakout', 'Resistance_Breakout', 'Support_Breakout']].any(axis=1)

    return df

def export_breakouts(df, filename='breakouts.csv'):
    """
    Exports the breakout events to a CSV file and prints the breakout times.

    Parameters:
        df (pd.DataFrame): DataFrame containing breakout events.
        filename (str): Name of the output CSV file.
    """
    breakout_events = df[df['Any_Breakout']].copy()
    breakout_events = breakout_events[[
        '1min_open', '1min_high', '1min_low', '1min_close', '1min_volume',
        'Upward_Breakout', 'Downward_Breakout', 'Reversal_Breakout',
        'Resistance_Breakout', 'Support_Breakout'
    ]]
    breakout_events.to_csv(filename)
    print(f"Breakout events exported to {filename}")
    
    # Print the breakout times and details
    print("\nBreakout Times and Details:")
    for index, row in breakout_events.iterrows():
        print(f"Time: {index}, Open: {row['1min_open']}, High: {row['1min_high']}, Low: {row['1min_low']}, Close: {row['1min_close']}, Volume: {row['1min_volume']}")
        if row['Upward_Breakout']:
            print("  -> Upward Breakout")
        if row['Downward_Breakout']:
            print("  -> Downward Breakout")
        if row['Reversal_Breakout']:
            print("  -> Reversal Breakout")
        if row['Resistance_Breakout']:
            print("  -> Resistance Breakout")
        if row['Support_Breakout']:
            print("  -> Support Breakout")

    # Count and print the number of breakouts
    print("\nNumber of Breakouts:")
    print(f"Upward Breakouts: {breakout_events['Upward_Breakout'].sum()}")
    print(f"Downward Breakouts: {breakout_events['Downward_Breakout'].sum()}")
    print(f"Reversal Breakouts: {breakout_events['Reversal_Breakout'].sum()}")
    print(f"Resistance Breakouts: {breakout_events['Resistance_Breakout'].sum()}")
    print(f"Support Breakouts: {breakout_events['Support_Breakout'].sum()}")

def main():
    # Use the provided API key
    API_KEY = 'Z546U0RSBDK86YYE'
    SYMBOL = 'SPY'  # Stock symbol to analyze
    INTERVALS = ['1min']  # Intervals to fetch data for
    OUTPUT_SIZE = 'full'  # 'compact' or 'full'

    # Fetch Data
    print("Fetching data from Alpha Vantage...")
    data_frames = []
    for interval in INTERVALS:
        try:
            df = get_intraday_data(SYMBOL, interval, API_KEY, OUTPUT_SIZE)
            data_frames.append(df)
            print(f"Data for interval {interval} fetched successfully.")
            time.sleep(12)  # To respect API rate limits
        except Exception as e:
            print(f"Error fetching data for interval {interval}: {e}")

    # Combine DataFrames
    if not data_frames:
        print("No data fetched. Exiting.")
        return
    data = pd.concat(data_frames, axis=1)
    print("Data combined successfully.")

    # Add Technical Indicators
    data = add_technical_indicators(data)
    print("Technical indicators added.")

    # Identify Breakouts
    data = identify_breakouts(data)
    print("Breakout events identified.")

    # Export Breakouts and print times and counts
    export_breakouts(data, filename=f'{SYMBOL}_breakouts.csv')

if __name__ == "__main__":
    main()
