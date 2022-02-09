# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:42:04 2022

@author: Bruno M. Breggia
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from random_forest import RandomForest

accuracy = lambda y_true, y_pred: np.sum(y_true==y_pred) / len(y_true)

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

classifier = RandomForest(n_trees=3) # optimize code so it can manage quick training of more trees
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Random Forest accuracy is", accuracy(y_test, y_pred))


