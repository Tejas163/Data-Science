# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:35:20 2018
Blood transfusion classification problem
@author: Tejas
"""
##import all the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
##read the data files
df=pd.read_csv('E:/Books/Data Science/project-1-blood transfusion/datasets/blood-transfusion.csv')
##read the data
df.head()
df.count()
##check the datatype
print("the keys of the data is:\n{}".format(df.keys()))
##check the features
##print("the features of the data are:{}".format(df['feature_names']))
##check the size of data
df.shape
##store the target variable as target
file=df.drop('target', axis=1).values
df1=df['target'].values
file.shape
file.dtype
##check for null and nan values
pd.isnull(df).any()
np.isnan(df).any()
#divide into training and testing set
f_train,f_test,x_train,x_test=train_test_split(file,df1,random_state=42)
##check the shape of training and testing set
print("the shape of training set:{}".format(f_train.shape))
print("the shape of test set:{}".format(f_test.shape))
x_train.shape
x_test.shape
## train and fit the algorithms and predict
clf=tree.DecisionTreeClassifier()
clf.fit(f_train,x_train)
clf.score(f_test,x_test)
pred=clf.predict(f_test)
##Using svm
svclf=SVC(kernel="linear", C=1)
svclf.fit(f_train,x_train)
svclf.score(f_test,x_test)
pred1=svclf.predict(f_test)

#print the results
print("Test set prediction using decision trees:{}".format(pred))
print("Test set predictions using svm's:{}".format(pred1))
##Get the scores of the rest
print("The accuracy score is:",accuracy_score(pred,x_test))
print("the accuracy score of using SVM is:",accuracy_score(pred1,x_test))
##accuracy_score(pred1,f_test)

