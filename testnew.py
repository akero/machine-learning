from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

try:
    model = load_model('stock_prediction_model29.h5')
except:
    print("Error loading model. Please make sure the file 'stock_prediction_model.h5' exists.")
    exit()

#load the new data

try:
    new_data = pd.read_csv('test_data.csv')
except:
    print("Error reading test data file. Please make sure the file 'test_data.csv' exists and is valid.")
    exit()

    
    
new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].values


# Step 3: Load 30 rows of the latest price continuing from the end of test_data.csv
# 
try:
    data = pd.read_csv('Apple-real.csv')
except:
    print("Error reading original data file. Please make sure the file 'Apple.csv' exists and is valid.")
    exit()

date_close = data[['Date', 'Close']]

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data[['Close']])

# Step 4: Load the scaler and scale the new data
scaler = MinMaxScaler(feature_range=(0, 1))
new_data = scaler.fit_transform(new_data)

# Step 5: Reshape the new data into the shape expected by the model
lookback = 15
X = []
for i in range(lookback, len(new_data)):
    X.append(new_data[i-lookback:i])
X = np.array(X)


# Step 6: Use your trained model to make predictions with 16 day limit
predictions = model.predict(X)[:16]

# Step 7: Transform the predicted values back to their original scale
predictions = close_scaler.inverse_transform(predictions)

print(predictions.min(), predictions.max())

# Step 8: Visualize the predictions and the last 30 days of the actual data

# Assuming you have already split your data into training and testing sets
# and have made predictions on the test set using your model
test_set = pd.read_csv('Apple-real.csv')

y_test = close_scaler.inverse_transform(test_set['Close'].values.reshape(-1, 1))
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