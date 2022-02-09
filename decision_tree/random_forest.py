# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:21:39 2022

@author: Bruno M. Breggia
"""

import numpy as np
from collections import Counter
from decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, 
                                self.max_depth, 
                                self.n_features)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array( [tree.predict(X) for tree in self.trees] )
        # [1111 0000 1111]
        # [101 101 101 101]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

