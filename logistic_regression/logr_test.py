# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:59:44 2022

@author: Bruno M. Breggia
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], y, color='b', marker='o', s=30)
plt.show()

# accuracy
accuracy = lambda y_true, y_pred: np.sum(y_true == y_pred) / len(y_true)

from logistic_regression import LogisticRegression

regressor= LogisticRegression(learning_rate=0.0001, iterations=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print('LR classification accuracy:', accuracy(y_test, predictions))
