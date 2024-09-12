import requests

# Your Polygon.io API key
api_key = "LN7rpB3UFWprPPOTQhVlXXtqA2Xp7NRg"

def get_available_options(symbol, limit=1):
    url = f"https://api.polygon.io/v3/reference/options/contracts"
    params = {
        'underlying_ticker': symbol,
        'apiKey': api_key,
        'order': 'asc',  # Sort by earliest date first
        'limit': limit,  # Limit results to check the earliest contracts
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            earliest_contract = data['results'][0]
            ticker = earliest_contract['ticker']
            expiration_date = earliest_contract['expiration_date']
            return f"Earliest option contract for {symbol} is {ticker} with expiration date {expiration_date}."
        else:
            return "No option data available for this symbol."
    else:
        return f"Error: {response.status_code} - {response.text}"

symbol = "SPY"  # Replace with your desired symbol
earliest_data = get_available_options(symbol)
print(earliest_data)
