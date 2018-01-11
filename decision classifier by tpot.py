# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:23:00 2018

@author: Tejas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.metrics import accuracy_score
# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('<FILE_PATH>', sep=',', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.8074589127686472
exported_pipeline = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=BernoulliNB(alpha=10.0, fit_prior=False)),
    DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=5, min_samples_split=15)
)

exp1=exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
accuracy_score(exp1,results)
