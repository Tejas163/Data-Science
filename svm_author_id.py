#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
##decrease the training set and test set size to 1/3
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)//100]
#labels_train = labels_train[:len(labels_train)//100]
clf=SVC(kernel="rbf",C=10000.0)
clf
t0=time()
clf.fit(features_train,labels_train)
train_time=time()-t0
print("train time is : %0.3fs" %train_time)
print("trainnig score:{}".format(clf.score(features_train,labels_train)))
t0=time()
pred=clf.predict(features_test)
test_time=time()-t0
print("test time is : %0.3fs" %test_time)
score=accuracy_score(labels_test,pred)
print("The score is:%.3f" %score)
#answer=pred[1700]
#print(answer)
##find the number of chris'emails
#count = 0
#l=1

#for x in pred:
 #   if (pred[l]==1):
  #      count += 1
   #     l += 1
   # else:
   #     continue

#print(count)

print(Counter(pred))
#########################################################
### your code goes here ###

#########################################################


