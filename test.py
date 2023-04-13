from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the saved model
try:
    model = load_model('stock_prediction_model29.h5')
except:
    print("Error loading model. Please make sure the file 'stock_prediction_model.h5' exists.")
    exit()

# Load the new data that model will make predictions on
try:
    new_data = pd.read_csv('test_data.csv')
except:
    print("Error reading test data file. Please make sure the file 'test_data.csv' exists and is valid.")
    exit()

# Modify the new_data variable to only contain the relevant columns (excluding 'Close' and 'Adj Close')
new_data = new_data[['Open', 'High', 'Low', 'Volume']].values

# Load the original data and extract the last 30 rows for the date_close variable
try:
    data = pd.read_csv('Apple.csv')
except:
    print("Error reading original data file. Please make sure the file 'Apple.csv' exists and is valid.")
    exit()

date_close = data[['Date', 'Close']].tail(30)
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data[['Close']])

# Scale the new data
scaler = MinMaxScaler(feature_range=(0, 1))
new_data = scaler.fit_transform(new_data)

# Reshape the new data into the shape expected by the model
lookback = 15
X = []
for i in range(lookback, len(new_data)):
    X.append(new_data[i-lookback:i])
X = np.array(X)

# Use your trained model to make predictions
predictions = model.predict(X)

# Transform the predicted values back to their original scale
predictions = close_scaler.inverse_transform(predictions)

# Visualize the predictions and the last 30 days of the actual data
test_set = pd.read_csv('test_data.csv')
y_test = close_scaler.inverse_transform(date_close['Close'].values.reshape(-1, 1))
y_pred = close_scaler.inverse_transform(predictions)

# Plot the true stock prices
plt.plot(y_test, color='red', label='True Price')
# Plot the predicted stock prices
plt.plot(y_pred, color='blue', label='Predicted Price')
# Add labels and title
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# Show the plot
plt.show()