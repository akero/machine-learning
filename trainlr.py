import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import ParameterGrid
import math

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

# define the parameter grid for the grid search
param_grid = {
    'initial_lrate': [0.1, 0.01, 0.001],
    'drop': [0.5, 0.6, 0.7],
    'epochs_drop': [5.0, 10.0, 15.0]
}

# create a ParameterGrid object
grid = ParameterGrid(param_grid)

best_val_loss = float('inf')
best_params = None

# iterate over all combinations of parameter values
for params in grid:
    # set the parameter values
    initial_lrate = params['initial_lrate']
    drop = params['drop']
    epochs_drop = params['epochs_drop']
    
    # define the step decay function
    def step_decay(epoch):
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    # create a LearningRateScheduler callback
    lrate = LearningRateScheduler(step_decay)

    # compile and fit your model
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[lrate], verbose=0)
    
    # evaluate the performance of your model on the validation set
    val_loss = history.history['val_loss'][-1]
    
    # check if this is the best set of parameter values so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params

# print the best set of parameter values and the corresponding validation loss
print(f'Best parameters: {best_params}, Best validation loss: {best_val_loss}')

# Evaluate the model on the testing set using the best set of parameter values
initial_lrate = best_params['initial_lrate']
drop = best_params['drop']
epochs_drop = best_params['epochs_drop']

def step_decay(epoch):
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[lrate])


# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)

# Save the trained model
model.save('prediction16lrmodel.h5')