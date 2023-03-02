#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 08:01:19 2023

@author: Thibault RAGU
"""
print('LAB 4 : ')

import numpy as np
import matplotlib.pyplot as plt


###Part I : PCA through Singular Value Decomposition
print('Part I : PCA through Singular Value Decomposition')

# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])

# Calculate the covariance matrix:
X1 = np.transpose(X)        #X1 is the transpose of X
R = 1/3*np.matmul(X,X1)     #R is the covariance matrix of X

print('We calculate the covariance matrix : ', R)

# Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]

# Calculate the coordinates in new orthonormal basis:
Xi1 = np.matmul(np.transpose(X),u1)
Xi2 = np.matmul(np.transpose(X),u2)

print('We calculate the coordinates in new orthonormal basis : ')
print('For Xi1, we have : ', Xi1)
print('For Xi2, we have : ', Xi2)

# Calculate the approximation of the original from new basis
Xaprox = np.matmul(u1[:,None],Xi1[None,:]) 
print('We calculate the approximation of the original from new basis : ', Xaprox)


###Part 2 : PCA on Iris data
print('Part 2 : PCA on Iris data')
from sklearn.datasets import load_iris

# Load Iris dataset as in the last PC lab:
iris=load_iris()    #the IRIS dataset is stored in the iris variable
iris.feature_names  #the array contains the names of the four features in the Iris dataset. 

print('We can print out the names of the features : ', iris.feature_names)
print('We can print out the first five rows of the data matrix : ', iris.data[0:5,:])
print('#We can print out the first five values of the target variable : ', iris.target[:])

# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data      #assigns the feature data to the variable X
y=iris.target    #assigns the feature data to the variable y
axes1=plt.axes(projection='3d')        #creates a 3D plot
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show()       #displays the 3D plot

# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
#from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

#We define PCA object (three components), fit and transform the data
pca = PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())

#We plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show()

#We compute pca.explained_variance_ and pca.explained_cariance_ratio_values
pca.explained_variance_
pca.explained_variance_ratio_

#We plot the principal components in 2D, mark different targets in color
plt.scatter(Xpca[y==0,0],Xpca[y==0,1], color = 'green')
plt.scatter(Xpca[y==1,0],Xpca[y==1,1], color = 'blue')
plt.scatter(Xpca[y==2,0],Xpca[y==2,1], color = 'magenta')

###Part 3 : KNN classifier
print('Part 3 : KNN classifier')

# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(X_train.shape)
print(X_test.shape)
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_train)
Ypred=knn1.predict(X_test)

# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
var = confusion_matrix(y_test,Ypred)
ConfusionMat = ConfusionMatrixDisplay(confusion_matrix = var)
ConfusionMat.plot()
plt.show()

# Now do the same (data set split, KNN, confusion matrix), but for PCA-transformed data (1st two principal components, i.e., first two columns). 
# Compare the results with full dataset
X_trainWrong, X_testWrong, y_trainWrong, y_testWrong = train_test_split(X[:,0:1],y,test_size=0.3)
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_trainWrong,y_trainWrong)
YpredWrong=knn1.predict(X_testWrong)
var = confusion_matrix(y_testWrong,Ypred)
ConfusionMat = ConfusionMatrixDisplay(confusion_matrix = var)
ConfusionMat.plot()
plt.show()



