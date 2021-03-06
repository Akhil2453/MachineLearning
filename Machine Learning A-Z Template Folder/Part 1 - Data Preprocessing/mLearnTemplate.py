# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:44:45 2018

@author: Akhil
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values   #storing feature data(country,age,salary), in x
                                  #iloc[:, :-1] selects the first three columns of the dataset
y = dataset.iloc[:, 3].values     # storing dependent vector in y                               

#taking care of missing data
"""from sklearn.preprocessing import Imputer 
                    #sklearn includes methods to preprocess datasets.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])"""
#not necessary in all mLearning example

# Encoding categorical Data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
#Use onehot encoder, as there's no comparison between countries, and we don't want the algorithm to compare them
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
#use label encoder, as we have tocompare the purchases and it's dependent vector
labelencoder_y = LabelEncoder()
y = labelencoder_x.fit_transform(y)"""
#not necessary in all mLearning examples

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
x_train,  x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""
#not necessary in all example

