from pandas import read_csv
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

#Identifies Missing values and imputes the values with Mean Value
def Impute():
    dataset = read_csv('D:\Skillenza\DataScience-POC-Usecase-20200329T165548Z-001\DataScience-POC-Usecase\Small\Training.csv', header=0)
    print('Count of the number of Missing values in each column')
    print(dataset.isnull().sum())
    print('----------------------------------------------------------------------------')
    
    print('Replace NaN values with Mean')
    dataset.fillna(dataset.mean(), inplace=True)
    print('Data Imputed with Mean values')
    print(dataset[['Social_Search_Impressions','Social_Search_Working_cost']])

Impute()
