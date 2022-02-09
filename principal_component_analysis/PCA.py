# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 23:22:41 2022

@author: Bruno M. Breggia
"""

import numpy as np

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # mean 
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        
        # covariance
        cov = np.cov(X.T)
        
        # eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        
        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]
        
    def transform(self, X):
        # project data
        X -= self.mean
        return np.dot(X, self.components.T)
    
