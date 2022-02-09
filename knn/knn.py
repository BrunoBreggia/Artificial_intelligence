# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 21:38:41 2022

@author: Bruno M. Breggia
"""

import numpy as np
from collections import Counter

euclidean_distance = lambda x1,x2: np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict(self, x):
        predicted_labels = [self._predict(item) for item in x]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]
        
        # get k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        K_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # majority vote, most common label
        most_common = Counter(K_nearest_labels).most_common(1)
        return most_common[0][0]


