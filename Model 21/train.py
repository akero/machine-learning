import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

df = pd.read_csv('Apple.csv')

# Select relevant columns
data = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
traindata, testdata = data[0:train_size,:], data[train_size:len(data),:]

#Step 4: Create LSTM input sequences
#The LSTM model needs to receive input sequences of a specific length. You can create these input sequences using a sliding window approach. For example, if you want to use 60 days of historical data to predict the next day's stock price, you can create input sequences with a length of 60.

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), :])
        y.append(data[(i+seq_length), 3])
    return np.array(X), np.array(y)

seq_length = 15
X_train, y_train = create_sequences(traindata, seq_length)
X_test, y_test = create_sequences(testdata, seq_length)

#Step 5: Define and train the LSTM model
#Define the LSTM model using Keras. You can start with a simple model that has a single LSTM layer followed by a dense output layer. Then, train the model using the training set.

model = Sequential()
model.add(LSTM(350, input_shape=(seq_length, 6), return_sequences=True))
model.add(LSTM(350, return_sequences=True))
model.add(LSTM(350, return_sequences=True))
model.add(LSTM(350))
model.add(Dense(1))

# Set the desired learning rate
learning_rate = 0.0001

# Create an instance of the Adam optimizer with the desired learning rate
adam = Adam(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer='adam')

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=10)

# Train the model with the early stopping callback
history = model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)

# Save the trained model
model.save('stock_prediction_model21.h5')