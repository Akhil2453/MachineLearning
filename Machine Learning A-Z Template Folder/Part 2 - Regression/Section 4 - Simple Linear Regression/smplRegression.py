# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:39:45 2018

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

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
x_train,  x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""
#not necessary in all example

