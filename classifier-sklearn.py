# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:35:20 2018

@author: Tejas
"""
##import all the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
##read the data files
df=pd.read_csv('E:/Books/Data Science/project-1-blood transfusion/data/blood-transfusion.csv')
##read the data
df.head()
##check the datatype
print("the keys of the data is:\n{}".format(df.keys()))
##check the features
print("the features of the data are:{}".format(df['feature_names']))
##check the size of data
df.shape
##store the target variable as target
file=df.drop('target', axis=1).values
##check for null and nan values
pd.isnull(df).any()
np.isnan(df).any()
#divide into training and testing set
f_train,f_test,x_train,x_test=train_test_split(file,df,random_state=42)
##check the shape of training and testing set
print("the shape of training set:{}".format(f_train.shape))
print("the shape of test set:{}".format(f_test.shape))
df['data'].groupby




clf=tree.DecisionTreeClassifier()
svclf=SVC()
