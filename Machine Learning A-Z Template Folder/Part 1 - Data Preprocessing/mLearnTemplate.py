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
from sklearn.preprocessing import Imputer 
                    #sklearn includes methods to preprocess datasets.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)