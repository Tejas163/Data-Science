# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:35:20 2018
Blood transfusion classification problem
@author: Tejas
"""
##import all the packages
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC,LinearSVC
from sklearn.pipeline import Pipeline,FeatureUnion
##read the data files
df=pd.read_csv('E:/Books/Data Science/project-1-blood transfusion/datasets/blood-transfusion.csv')
##read the data
df.head()
df.info()
df.count()
df.describe()
list(df.columns)
display(df)

##check the datatype
print("the keys of the data is:\n{}".format(df.keys()))
##check the features
##print("the features of the data are:{}".format(df['feature_names']))
##plot the data
df.hist(bins=50,figsize=(20,15))
plt.show()
##feature selection
#scaler=StandardScaler()
#scaler_d=scaler.fit_transform(df)
##store the target variable as target
#file=df.drop('target', axis=1).values
#file.keys()
#df1=df['target'].values
#file.shape
#file.dtype
#df1.shape
##check for null and nan values
pd.isnull(df).any()
np.isnan(df).any()
#divide into training and testing set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in split.split(df, df["target"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
#X_train,X_test=train_test_split(df,train_size=0.6,random_state=42)
##check the shape of training and testing set
#print("the shape of training set:{}".format(x_train.shape))
#print("the shape of test set:{}".format(x_test.shape))
df["target"].value_counts() / len(df)
#copy the data 
df=strat_train_set.copy()
##plot the graphs
df.plot(kind="scatter", x="Monetary (c.c. blood)", y="target", alpha=0.1)
df.plot(kind="scatter", x="Recency (months)", y="target", alpha=0.1)
df.plot(kind="scatter", x="Frequency (times)", y="target", alpha=0.1)
df.plot(kind="scatter", x="Recency (months)", y="Frequency (times)", alpha=0.1)
##Look for correlations
corr_matrix=df.corr()
corr_matrix["Frequency (times)"].sort_values(ascending=False)
#scatter matrix
attributes = ["Frequency (times)", "Monetary (c.c. blood)", "Recency (months)"]
scatter_matrix(df[attributes], figsize=(12, 8))

#separare predictors and labels
df = strat_train_set.drop("target", axis=1)
df_labels = strat_train_set["target"].copy()
df1=strat_test_set.drop("target", axis=1)
df1_labels=strat_test_set["target"].copy()
## train and fit the algorithms and predict
##create a pipeline


##Custom transformer class
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
## create a pipeline
num_attribs = list(df)
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', Imputer(strategy="median")),('std_scaler', StandardScaler())])
df_new=num_pipeline.fit_transform(df)
## testing set
num_attribs = list(df1)
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', Imputer(strategy="median")),('std_scaler', StandardScaler())])
df1_new=num_pipeline.fit_transform(df1)

#1-Random Forest on training set
clf=RandomForestClassifier()
clf.fit(df_new,df_labels)
df_predictions=clf.predict(df_new)
clf1_predictions=clf.predict(df1_new)
tree_mse = mean_squared_error(df_labels, df_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
##compute cross validation score
scores = cross_val_score(clf, df_new, df_labels,
scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
##display scores

clf.score(df_new,df_labels)
clf.score(df1_new,df1_labels)


#np.reshape(clf, (-1,1))
##Using svm
svclf=SVC(kernel="linear", C=1)
svclf.fit(df_new,df_labels)
svclf.score(df_new,df_labels)
pred1=svclf.predict(df1_new)
svclf.score(df1_new,df1_labels)
#np.reshape(svclf, (-1,1))
##Using decision trees
dtclf=tree.DecisionTreeClassifier(random_state=2)
dtclf.fit(df_new,df_labels)
dtclf.score(df_new,df_labels)
pred2=dtclf.predict(df_new)
pred2=dtclf.predict(df1_new)  ##Test set
#print the results
print("Test set prediction using random forest trees:{}".format(pred))
print("Test set predictions using svm's:{}".format(pred1))
print("Test set prediction using decision trees:{}".format(pred2))
##Get the scores of the rest
print("The accuracy score of random forest is:",accuracy_score(pred,df1_labels))
print("the accuracy score of using SVM is:",accuracy_score(pred1,df1_labels))
print("the accuracy score of using decision tree classifier is:",accuracy_score(pred2,df1_labels))
##accuracy_score(pred1,f_test)

