                                                                                   # Import necessary libraries
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import subprocess

# Function to check GPU usage
def check_gpu_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv'])
        print(result.decode('utf-8'))
    except Exception as e:
        print(f"Error checking GPU usage: {e}")

# Example data
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 3, 4, 5, 6],
    'target': [5, 7, 9, 11, 13]
})

# Features and target variable
X = data[['feature1', 'feature2']]
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBRegressor with GPU support and verbose logging
model = XGBRegressor(tree_method='hist', device='cuda', verbosity=2)
model.fit(X_train, y_train)

# Check GPU usage
check_gpu_usage()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Check GPU usage again
check_gpu_usage()
