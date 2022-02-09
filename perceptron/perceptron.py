# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 18:07:33 2022

@author: Bruno M. Breggia
"""

import numpy as np

class Perceptron:
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None
        
    def _unit_step_function(self, x):
        return np.where(x>=0, 1, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y_ = np.array([1 if i>0 else 0 for i in y])
        
        for _ in range(self.iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
   
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    
    
    