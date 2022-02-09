# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:00:33 2022

@author: Bruno M. Breggia
"""

import numpy as np

class DecisionStump:
    
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alfa = None
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions
    
    
class Adaboost:
    
    def __init__(self, n_classifiers=5):
        self.n_classifiers = n_classifiers
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # init weights
        w = np.full(n_samples, 1/n_samples)
        self.clfs = []
        for _ in range(self.n_classifiers):
            clf = DecisionStump()
            
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i] 
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    pol = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    
                    missclasified = w[ y != predictions ]
                    error = sum(missclasified)
                    
                    if error > 0.5:
                        pol = -1
                        error = 1 - error
                    
                    if error < min_error:
                        min_error = error
                        clf.polarity = pol
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
            EPS = 1e-10 # epsilon de maquina
            clf.alpha = 0.5*np.log( (1-error)/(error+EPS) )
            predictions = clf.predict(X)
            
            w *= np.exp(-clf.alpha*y*predictions)
            w /= sum(w)
            
            self.clfs.append(clf)
    
    def predict(self, X):
        clf_preds = [clf.alpha*clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
        
        
