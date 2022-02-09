# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:47:22 2022

@author: Bruno M. Breggia
"""

import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.iterations):
            
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1/n_samples) * np.dot(X.T, y_predicted - y)
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db
            
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    
