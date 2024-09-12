import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Fetch data from Alpha Vantage
api_key = 'Z546U0RSBDK86YYE'
symbol = 'TSLA'

ts = TimeSeries(key=api_key, output_format='pandas')

# Define date range for 5 years
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Fetching intraday data (with compact size)
data_1min, _ = ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
data_5min, _ = ts.get_intraday(symbol=symbol, interval='5min', outputsize='compact')
data_15min, _ = ts.get_intraday(symbol=symbol, interval='15min', outputsize='compact')
data_30min, _ = ts.get_intraday(symbol=symbol, interval='30min', outputsize='compact')
data_1hour, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize='compact')

# Filter data to only include the last 5 years
data_1min = data_1min[data_1min.index >= start_date]
data_5min = data_5min[data_5min.index >= start_date]
data_15min = data_15min[data_15min.index >= start_date]
data_30min = data_30min[data_30min.index >= start_date]
data_1hour = data_1hour[data_1hour.index >= start_date]

# Concatenate all intraday data into a single DataFrame
data_combined = pd.concat([data_1min, data_5min, data_15min, data_30min, data_1hour])

# Reset index to have a single datetime index
data_combined = data_combined.reset_index()
data_combined = data_combined.rename(columns={'date': 'datetime'})

# Feature Engineering
data_combined['previous_close'] = data_combined['4. close'].shift(1)
data_combined['price_change'] = data_combined['4. close'] - data_combined['1. open']
data_combined['ma5'] = data_combined['4. close'].rolling(window=5, min_periods=1).mean()
data_combined['ma10'] = data_combined['4. close'].rolling(window=10, min_periods=1).mean()
data_combined['ma20'] = data_combined['4. close'].rolling(window=20, min_periods=1).mean()
data_combined['vol_change'] = data_combined['5. volume'].pct_change().fillna(0)
data_combined['high_low_diff'] = data_combined['2. high'] - data_combined['3. low']
data_combined['open_close_diff'] = data_combined['1. open'] - data_combined['4. close']

# Add more technical indicators
data_combined['ema5'] = data_combined['4. close'].ewm(span=5, adjust=False).mean()
data_combined['ema20'] = data_combined['4. close'].ewm(span=20, adjust=False).mean()
data_combined['momentum'] = data_combined['4. close'] - data_combined['4. close'].shift(4).fillna(0)
data_combined['volatility'] = data_combined['4. close'].rolling(window=5, min_periods=1).std()

# Add additional features
data_combined['roc'] = data_combined['4. close'].pct_change(periods=10)  # Rate of change
data_combined['ema12'] = data_combined['4. close'].ewm(span=12, adjust=False).mean()
data_combined['ema26'] = data_combined['4. close'].ewm(span=26, adjust=False).mean()
data_combined['macd'] = data_combined['ema12'] - data_combined['ema26']

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data_combined['rsi'] = calculate_rsi(data_combined['4. close'])

# Add lag features
lags = 30
lag_columns = ['4. close', '2. high', '3. low', '5. volume']
lagged_data = pd.concat([data_combined[lag_columns].shift(i).add_suffix(f'_lag_{i}') for i in range(1, lags+1)], axis=1)
data_combined = pd.concat([data_combined, lagged_data], axis=1)

# Drop rows with any missing values after adding lag features
data_combined = data_combined.dropna()

# Replace infinite values and very large values
data_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
data_combined.fillna(0, inplace=True)

# Define features and labels
features = [
    'previous_close', '1. open', 'ma5', 'ma10', 'ma20', 'vol_change', 'high_low_diff', 'open_close_diff',
    'ema5', 'ema20', 'momentum', 'volatility', 'roc', 'macd', 'rsi'
] + [f'{col}_lag_{i}' for i in range(1, lags+1) for col in lag_columns]

X = data_combined[features].values
y = data_combined[['2. high', '3. low']].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=8, num_layers=3, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.transformer = nn.Transformer(
            d_model=dim_feedforward,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        self.fc = nn.Linear(dim_feedforward, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src, src)
        output = self.fc(output[-1, :, :])  # Use the last output for prediction
        return output

# Initialize and train the model
model = TransformerModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Calculate accuracy metrics
mape_high = mean_absolute_percentage_error(y_test[:, 0], predictions[:, 0]) * 100
mape_low = mean_absolute_percentage_error(y_test[:, 1], predictions[:, 1]) * 100
mse_high = mean_squared_error(y_test[:, 0], predictions[:, 0])
mse_low = mean_squared_error(y_test[:, 1], predictions[:, 1])

# Print accuracy metrics
print(f"High Prediction MAPE: {mape_high:.2f}%")
print(f"Low Prediction MAPE: {mape_low:.2f}%")
print(f"High Prediction MSE: {mse_high:.2f}")
print(f"Low Prediction MSE: {mse_low:.2f}")

# Output actual and predicted values for comparison
comparison = pd.DataFrame({
    'Datetime': data_combined['datetime'][y_test.index],
    'Actual_High': y_test[:, 0],
    'Predicted_High': predictions[:, 0],
    'Actual_Low': y_test[:, 1],
    'Predicted_Low': predictions[:, 1]
})

# Plot the actual and predicted High and Low prices
plt.figure(figsize=(14, 5))
plt.scatter(comparison['Datetime'], comparison['Actual_High'], color='red', label='Actual High Price')
plt.scatter(comparison['Datetime'], comparison['Predicted_High'], color='blue', label='Predicted High Price')
plt.scatter(comparison['Datetime'], comparison['Actual_Low'], color='green', label='Actual Low Price')
plt.scatter(comparison['Datetime'], comparison['Predicted_Low'], color='orange', label='Predicted Low Price')

# Sort by Datetime to ensure lines are drawn in the correct order
comparison_sorted = comparison.sort_values(by='Datetime')

# Draw lines connecting the points by the closest dates
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Actual_High'], color='red', alpha=0.5, linestyle='-')
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Predicted_High'], color='blue', alpha=0.5, linestyle='-')
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Actual_Low'], color='green', alpha=0.5, linestyle='-')
plt.plot(comparison_sorted['Datetime'], comparison_sorted['Predicted_Low'], color='orange', alpha=0.5, linestyle='-')

plt.title(f'{symbol} High and Low Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
