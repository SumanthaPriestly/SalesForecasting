import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.metrics import accuracy_score
from datetime import datetime
from numpy import array
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


#Load Training Data
df = read_csv('D:\Skillenza\DataScience-POC-Usecase-20200329T165548Z-001\DataScience-POC-Usecase\Small\Training.csv', header=0)
dat = df.iloc[:, 1:2].values

#Load Testing Data
t_df = read_csv('D:\Skillenza\DataScience-POC-Usecase-20200329T165548Z-001\DataScience-POC-Usecase\Small\Test.csv', header=0)
t_dat = t_df.iloc[:, 1:2].values

# define input sequence
train = dat
test = t_dat


model = ExponentialSmoothing(train, trend="add",seasonal_periods=13)
fit = model.fit()
pred = fit.forecast(26)


final = pred[[0,5,13,18,23]]

plt.figure(figsize=(10,6))
plt.plot(test, color='blue', label='Actual')
plt.plot(final , color='red', label='Predicted')
plt.title('Sales Forecasting')
plt.xlabel('Forecast Horizons (Day)')
plt.ylabel('Sales')
plt.legend()
plt.show()

print('-------------------------------------------------------')
print('MAE:')
print(metrics.mean_absolute_error(test,final))
print('')

print('RMSE:')
print(np.sqrt(metrics.mean_absolute_error(test,final)))

print('')
print('')
print('-------------------------------------------------------')