# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:30:51 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.metrics import accuracy_score
from numpy import array
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#Load Training Data
df = read_csv("/Users/ange/Downloads/Training-Data-Sets.csv", header=0)
dat = df.iloc[:, 1:2].values

#Load Testing Data
t_df = read_csv("/Users/ange/Downloads/Test dataset v1.csv", header=0)
t_dat = t_df.iloc[:, 1:2].values

train = dat
test = t_dat
print (len(test))


model = ExponentialSmoothing(train, trend="add" ,seasonal="add" ,seasonal_periods=13)
fit = model.fit()
pred = fit.forecast(39)

mae=metrics.mean_absolute_error(test,pred)
rmse=np.sqrt(metrics.mean_absolute_error(test,pred))


plt.figure(figsize=(10,6))
plt.plot(test, color='blue', label='Actual')
plt.plot(pred , color='red', label='Predicted')
plt.title('Sales Forecasting')
plt.xlabel('Forecast Horizons (Day)')
plt.ylabel('Sales')
plt.legend()
plt.show()

print('-------------------------------------------------------')
print('MAE:')
print(metrics.mean_absolute_error(test,pred))
print('')

print('RMSE:')
print(np.sqrt(metrics.mean_absolute_error(test,pred)))

print('')
print('')
print('-------------------------------------------------------')