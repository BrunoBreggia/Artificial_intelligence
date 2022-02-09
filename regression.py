# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:49:38 2022

@author: Bruno M. Breggia
"""

import numpy as np

#%% Base regression class

class BaseRegression:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate=learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias= 0
        
        for _ in range(self.iterations):
            
            y_predicted = self._approximation(X, self.weights) + self.bias
            
            dw = (1/n_samples)*np.dot(X.T, y_predicted-y)
            db = (1/n_samples)*np.sum(y_predicted-y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
    def _approximation(self, X, w, b):
        raise NotImplementedError()
    
    def predict(self, X):
        return self._predict(X, self.weights, self.bias)
    
    def _predict(self, X, w, b):
        raise NotImplementedError()

#%% Linear regression

class LinearRegression(BaseRegression):
        
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b
    
    def _predict(self, X, w, b):
        return np.dot(X, w) + b
    
    
#%% Logistic Regression

class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)
        
    def _sigmoid(self, X):
        return 1 / (1 + np.eXp(-X))
    
    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    





