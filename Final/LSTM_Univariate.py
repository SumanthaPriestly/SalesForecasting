# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:47:16 2020

@author: Admin
"""


import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn import metrics
from sklearn.metrics import accuracy_score
from datetime import datetime
from numpy import array
from keras.layers import Bidirectional
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end = i + n_steps
		# check if we are beyond the sequence
		if end > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end], sequence[end]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def LSTM_Univariate():
    #Load Training Data
    df = read_csv("/Users/ange/Downloads/Training-Data-Sets.csv", header=0)
    dat = df.iloc[:, 1:2].values
    
    #Load Testing Data
    t_df = read_csv("/Users/ange/Downloads/Test dataset v1.csv", header=0)
    t_dat = t_df.iloc[:, 1:2].values

    # define input sequence
    train = dat
    test = t_dat[13:] #chane

    # choose a number of time steps
    n_steps = 13
    # split into samples
    X, y = split_sequence(train, n_steps)
    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    print(X.shape, y.shape)

    # define model
    start_time = time.time()
    model = Sequential()
    model.add(Bidirectional(LSTM(25, activation='relu',return_sequences=True), input_shape=(n_steps, n_features) ))

    model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(n_steps, n_features) ))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=100, verbose=0, shuffle = False)

    print('')
    print('Prediction')
    # demonstrate prediction
    #Testing

    test_inputs = t_dat
    test_features = []
    for i in range(13,len(t_dat)):
        test_features.append(test_inputs[i-13:i, 0])
    test_features = np.array(test_features)

    print(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    print('Features - Shape')
    print(test_features.shape[1])
    print(test_features)

    x_input = array(test_features)
    print('')
    print(x_input.shape)

    predictions = model.predict(test_features, verbose=0)

    print('')
    print("Execution Time: %s seconds" % (time.time() - start_time))
    print('')

    actual = test


    plt.figure(figsize=(10,6))
    plt.plot(actual, color='blue', label='Actual Precipitation')
    plt.plot(predictions , color='red', label='Predicted Precipitation')
    plt.title('Sales Forecasting')
    plt.xlabel('Forecast Horizons (Day)')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    print('-------------------------------------------------------')
    print('MAE:')
    print(metrics.mean_absolute_error(actual,predictions))
    print('')

    print('RMSE:')
    print(np.sqrt(metrics.mean_absolute_error(actual,predictions)))
    print('Epochs: 100')
    model.summary()
    print('')
    print('')
    print('-------------------------------------------------------')
    
LSTM_Univariate()