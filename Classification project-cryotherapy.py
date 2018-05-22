# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:15:10 2018

@author: Tejas
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#importing TPOT
from tpot import TPOTClassifier

df=pd.read_excel("<PATH>\folder\cryotherapy.xlsx")
df.describe()
df.head()
y=df['Result_of_Treatment']
y.head()
X=df.drop('Result_of_Treatment',axis=1)
X.shape,y.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
X_test.shape
y_train.shape

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled.shape
clf=LogisticRegression()
svmclf=SVC(kernel='rbf')
rfclf=RandomForestClassifier()
tpotclf=TPOTClassifier()
model1=clf.fit(X_train,y_train)
model2=svmclf.fit(X_train,y_train)
model3=rfclf.fit(X_train,y_train)
model_auto_clf=tpotclf.fit(X_train)
score=cross_val_score(clf,X_train,y_train)
score2=cross_val_score(svmclf,X_train,y_train)
score3=cross_val_score(rfclf,X_train,y_train)
print("score is:%.2f\n,",score3)

##Tpot classifier
tpotclf=TPOTClassifier(generations=5,cv=5)
model_tpot_clf=tpotclf.fit(X_train,y_train)
score=tpotclf.score(X_test,y_test)
print(score)
tpotclf.export('classifier-pipeline.py')
#predict for test
y_pred=clf.predict(X_test)
y1_pred=svmclf.predict(X_test)
y2_pred=rfclf.predict(X_test)
score_1=accuracy_score(y_test,y_pred)
score_2=accuracy_score(y_test,y1_pred)
score_3=accuracy_score(y_test,y2_pred)
print("score is:",score_1,score_2,score_3)