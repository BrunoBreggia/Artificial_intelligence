# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 19:16:36 2022

@author: Bruno M. Breggia
"""

import numpy as np

class LinearRegression:
    
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate=learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.normalization_limit = 1000
    
    def fit(self, X, y, normalize=False):
        
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias= 0
        
        # normalization of features
        if n_features >= self.normalization_limit or normalize:
            X = self.normalize(X)
        
        # iterations (learning process)
        for _ in range(self.iterations):
            
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples)*np.dot(X.T, y_predicted-y)
            db = (1/n_samples)*np.sum(y_predicted-y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
    
    def normalize(self, X):
        """
        Normalizes the feature columns of X
        """
        
        for feature in X.T:
            f_mean = np.mean(feature)
            f_range = np.amax(feature) - np.amin(feature)
        
            feature -= f_mean
            feature /= f_range
            
        return X

