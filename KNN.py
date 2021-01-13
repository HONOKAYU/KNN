#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:25:37 2020
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv

@author: yutianchen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
%matplotlib inline

# load data from csv file
import wget
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
file_name = wget.filename_from_url(url)
print(file_name)

df = pd.read_csv(url)
df.head()

# data visualization and analysis
df['custcat'].value_counts()
df.hist(columns= 'income', bins = 50)


# feature set 
# define feature sets and convert pandas dataframe to numpy array:
df.columns
# astype(float)
x = df[['region', 'tenure', 'age','marital', 'address','income','ed', 'employ','retire','gender','reside' ]] .values
x[0:5]
# labels:
y = df['custcat'].values
y[0:5]

# normarlize data
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x[0:5]

# train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print("Train Set: ", x_train.shape, y_train.shape)
print("Test Set: ", x_test.shape, y_test.shape)

# applying KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
# training
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
neigh
# predicting
yhat = neigh.predict(x_test)
yhat[0:5]

# accuracy evaluation

from sklearn import metrics
print("Train Set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
print("Test Set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# when k=6 
k=6
neigh6 = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
yhat6 = neigh6.predict(x_test)
print("Train Set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(x_train)))
print("Test Set Accuracy: ", metrics.accuracy_score(y_test, yhat6))

# then, we calculate different accuracy of KNN for Ks
Ks = 1000
mean_acc = np.zeros(Ks-1)
std_acc = np.zeros(Ks-1)

for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    yhat = neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    
mean_acc
                   
# plot model accuracy for different numbers of neighbors
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc, mean_acc + 1* std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc, mean_acc + 3* std_acc, alpha=0.10, color = 'green')
plt.legend(('Accuracy', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print("the best accuracy was with", mean_acc.max(), "with k=",mean_acc.argmax()+1)














