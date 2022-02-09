# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:36:43 2022

@author: Bruno M. Breggia
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from adaboost import Adaboost

accuracy = lambda y_true, y_pred: np.sum(y_true==y_pred) / len(y_true)

data = datasets.load_breast_cancer()
X = data.data
y = data.target

y[y==0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Adaboost classification with 5 weak classifiers
clf = Adaboost(n_classifiers=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_test, y_pred)
print('Accuracy of adaboost:', acc)

