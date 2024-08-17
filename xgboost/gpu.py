import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import psutil
import GPUtil

# Function to monitor GPU usage
def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}, Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB, Load: {gpu.load*100}%")

# Function to monitor CPU usage
def print_cpu_usage():
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")

# Generate a large synthetic dataset
n_samples = 1_000_000
n_features = 50
X, y = np.random.rand(n_samples, n_features), np.random.randint(0, 2, size=n_samples)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset into DMatrix which is the data structure that XGBoost uses
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up parameters for XGBoost with GPU support
params = {
    'objective': 'binary:logistic',
    'max_depth': 10,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'tree_method': 'gpu_hist',  # This enables GPU acceleration
    'predictor': 'gpu_predictor',  # This ensures that prediction is done on the GPU as well
    'eval_metric': 'logloss'
}

# Check if GPU is set correctly
if params['tree_method'] == 'gpu_hist' and params['predictor'] == 'gpu_predictor':
    print("Running on GPU.")
else:
    print("Running on CPU.")

# Monitor CPU and GPU usage before training
print("Initial resource usage:")
print_cpu_usage()
print_gpu_usage()

# Start training
start_time = time.time()
bst = xgb.train(params, dtrain, num_boost_round=1000)
end_time = time.time()

# Monitor CPU and GPU usage after training
print("\nResource usage during training:")
print_cpu_usage()
print_gpu_usage()

# Predict using the trained model
y_pred = bst.predict(dtest)
y_pred = [1 if i > 0.5 else 0 for i in y_pred]

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy * 100:.2f}%')

# Print training time
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Monitor final CPU and GPU usage
print("\nFinal resource usage:")
print_cpu_usage()
print_gpu_usage()
