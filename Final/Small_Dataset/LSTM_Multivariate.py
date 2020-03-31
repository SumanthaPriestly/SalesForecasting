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
from sklearn import preprocessing


# split a multivariate sequence into samples
def split_sequence(sequence, target, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end = i + n_steps
		# check if we are beyond the sequence
		if end > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end], target[end]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def LSTM_Multivariate():
    dataset = read_csv('D:\Skillenza\DataScience-POC-Usecase-20200329T165548Z-001\DataScience-POC-Usecase\Small\Training.csv', header=0)

    print('----------------------------------------------------------------------------')
    print('Replace NaN values with Mean')
    dataset.fillna(dataset.mean(), inplace=True)
    print('----------------------------------------------------------------------------')
    
    values = dataset.values
    target = dataset.iloc[:, 1:2].values
    dat = dataset.iloc[:, [1,12,31]].values


    t_df = read_csv('D:\Skillenza\DataScience-POC-Usecase-20200329T165548Z-001\DataScience-POC-Usecase\Small\Test.csv', header=0)
    t_df.fillna(t_df.mean(), inplace=True)
    t_dat = t_df.iloc[:, [1,12,31]].values
    t_target = t_df.iloc[:, 1:2].values

    # define input sequence
    train = dat
    test = t_dat
    target_train = target
    target_test = t_target[1:]

    # choose a number of time steps
    n_steps = 1
    # split into samples
    X, y = split_sequence(train,target, n_steps)
    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])


    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 3
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    print(X.shape, y.shape)

    # define model
    start_time = time.time()
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features) ))

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
    for i in range(n_steps, len(test_inputs)):
        test_features.append(test_inputs[i-n_steps:i, 0:n_features])

    
    test_features = np.array(test_features)

    print(test_features[0])
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], n_features))
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
    actual = target_test

    plt.figure(figsize=(10,6))
    plt.plot(actual, color='blue', label='Actual Forecast')
    plt.plot(predictions , color='red', label='Predicted Forecast')
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
    print('')
    print('')
    print('-------------------------------------------------------')

LSTM_Multivariate()