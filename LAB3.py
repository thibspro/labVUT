#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:02:05 2023

@author: Thibault RAGU
"""
print('LAB 3 : ')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

###Part I : SVM for classification
print('Part I : SVM for classification')

#We load IRIS dataset 
from sklearn.datasets import load_iris
#We split data into training and testing parts:
from sklearn.model_selection import train_test_split


iris=load_iris()    #the IRIS dataset is stored in the iris variable
iris.feature_names  #the array contains the names of the four features in the Iris dataset. 

print('We can print out the names of the features : ', iris.feature_names)
print('We can print out the first five rows of the data matrix : ', iris.data[0:5,:])
print('#We can print out the first five values of the target variable : ', iris.target[:])

#We assign the data matrix from the Iris dataset to the X variable and choose only first two features 
#We assign the data matrix from the Iris dataset to the y variable and eliminate iris.target =2 from the data
X=iris.data[iris.target!=2,0:2]
y=iris.target[iris.target!=2]

#We train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#The training matrix will have 120 rows and 4 columns
#The test matrix will have 30 rows and 4 columns
print(X_train.shape)
print(X_test.shape)

#We use a Support Vector Machine for classification:
SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
print(SVMmodel.score(X_test,y_test))
print(SVMmodel.get_params())

#We store the support vectors
supvectors=SVMmodel.support_vectors_
print(supvectors.shape)
print(X_train.shape)
print(y_train.shape)

print(supvectors)
#We plot scatterplots in different colors of targets 0 and 1 and check the separability of the classes:
plt.scatter(X[y==0,0],X[y==0,1],color='green')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
#plt.scatter(X[y==2,0],X[y==2,1],color='cyan')
plt.scatter(supvectors[:,0],supvectors[:,1],color='red',marker='+',s=50)

#Separating line coefficients:
W=SVMmodel.coef_
b=SVMmodel.intercept_
xgr=np.linspace(min(X[:,0]),max(X[:,0]),100)

print(W[:,0])
print(W[:,1])
print(b)
ygr=-W[:,0]/W[:,1]*xgr-b/W[:,1]
plt.scatter(xgr,ygr)





