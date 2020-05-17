#Import libs
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys

#Read data for training
data = pd.read_csv("data/problemAND.csv", header=None)

#Reading/handling data test
x_train = data.loc[:,0:1]
y_train = data.loc[:,2:2]

'''
#Fit only to the training data
scaler = StandardScaler()
scaler.fit(x_train)

#Transform the data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
'''

print(x_train, "\n", y_train)