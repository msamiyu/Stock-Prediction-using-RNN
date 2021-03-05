# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 16:31:34 2019

@author: talk2
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

JPM_stock=pd.read_csv("C:/Users/talk2/Desktop/Python/JPM.csv", 
                       index_col="Date", parse_dates=True)



JPM_stock=JPM_stock.loc[:,"Adj Close"]

JPM_stock=JPM_stock.as_matrix()
JPM_stock=JPM_stock.reshape(len(JPM_stock),1)

sc=MinMaxScaler(feature_range = (0,1))
newdata_scaled=sc.fit_transform(JPM_stock)

newdata_scaled[:21]

x_train=[]
y_train=[]

for i in range(30,2100):
    x_train.append(newdata_scaled[i-30:i,0])
    y_train.append(newdata_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#layer
model_JPM = Sequential()
model_JPM.add(LSTM(units=50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model_JPM.add(Dropout(0.1))
model_JPM.add(LSTM(units=50, return_sequences = True))
model_JPM.add(Dropout(0.1))
model_JPM.add(LSTM(units=50, return_sequences = True))
model_JPM.add(Dropout(0.1))
model_JPM.add(LSTM(units=50, return_sequences = True))
model_JPM.add(Dropout(0.1))
model_JPM.add(LSTM(units=50, return_sequences = True))
model_JPM.add(Dropout(0.1))8
model_JPM.add(LSTM(units=50, return_sequences = True))
model_JPM.add(Dropout(0.1))
model_JPM.add(LSTM(units=50, return_sequences = True))
model_JPM.add(Dropout(0.1))
model_JPM.add(LSTM(units=50, return_sequences = True))
model_JPM.add(Dropout(0.1))
model_JPM.add(LSTM(units=50))
model_JPM.add(Dropout(0.1))
model_JPM.add(Dense(units=1))
model_JPM.compile(optimizer ='adam', metrics = ['mse'], loss = 'mean_absolute_error')
#fitting(epoch)
model_JPM.fit(x_train, y_train, epochs = 200, batch_size = 62)

x_test=[]
y_test=[]

for i in range(2001, len(JPM_stock)):
    x_test.append(newdata_scaled[i-30:i,0])
    y_test.append(newdata_scaled[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

forecast_stock = model_JPM.predict(x_test)

fore = sc.inverse_transform(np.array(forecast_stock).reshape(-1,1))
actual = sc.inverse_transform(np.array(y_test).reshape(-1,1))


plt.plot(actual, color='red', label='Actual price')
plt.plot(fore, color='blue', label='forecasted price')
plt.title('JPM stock prediction')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.show()

