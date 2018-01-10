# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:27:51 2018

@author: Tejas
"""
###Importing the packages needed
import pandas as pd
from tpot import TPOTClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

###Read in the data file
input_file=pd.read_csv("E:/Books/Data Science/project-1-blood transfusion/data/blood-transfusion.csv")
input_file.head()
##Description
details=input_file.describe()
details
 ##shape of the data
 d_shape=input_file.shape
 d_shape
 
 #fine the datatypes of the set
 input_file.dtypes
##Check number of levels in the data variable having
 for cat in ['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)','whether he/she donated blood in March 2007']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, input_file[cat].unique().size))
    
##check for null values
    pd.isnull(input_file).any()
    np.isnan(input_file).any()
    
##create a variable to store the prediction
    don_blood_in_march=input_file['whether he/she donated blood in March 2007'].values
    file= input_file.drop('class', axis=1).values
 ##Aplit into test set and train set

training_features, testing_features, training_classes, testing_classes = train_test_split(file, input_file['class'].values, random_state=42)
training_features.shape
##
training_indices, validation_indices = training_indices, testing_indices = train_test_split(file1, stratify = don_blood_in_march, train_size=0.75, test_size=0.25) 
training_indices.size
validation_indices.size
##
 
##Use tpot to fit and predict
 tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=15)
tpot.fit(training_features,training_classes)

##find the score
tpot.score(input_file[training_classes], input_file.loc[training_classes, 'class'].values)

tpot.export('tpot_blood-transfusion_pipeline.py')

tpot_blood-transfusion_pipeline.py