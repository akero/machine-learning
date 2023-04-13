import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.layers import Dropout

df = pd.read_csv('Apple.csv')
data = df[['Open', 'High', 'Low', 'Volume']]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
traindata, testdata = data[0:train_size,:], data[train_size:len(data),:]

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

dropout_rate = 0.4
model = Sequential()
model.add(LSTM(350, input_shape=(seq_length, 4), return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(350, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(350, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(350))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

learning_rate = 0.0001
adam = Adam(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(X_train, y_train, epochs=10, batch_size=32)

y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)

model.save('stock_prediction_model29.h5')