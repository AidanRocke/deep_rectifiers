#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:26:05 2017

@author: aidanrocke
"""

import numpy as np

def variable_importance(trained_model,validate_X,validate_Y):
    """
        A method for calculating variable importance by introducing Gaussian noise. 

        input:
            trained_model

    """
    
    yhat = np.round(trained_model.predict(validate_X))
    
    # get boolean array of correctly mapped indices:
    boolean = np.array(yhat == validate_Y)
    
    X = validate_X[boolean[:,0]]
    Y = validate_Y[boolean[:,0]]
    
    N, M = np.shape(validate_X)
    
    importance = np.zeros(M)
    
    for i in range(M):
        
        mu = np.mean(X[:,i])
        sigma = np.std(X[:,i])
        
        Xhat = X
        
        Xhat[:,i] = np.random.normal(mu,sigma)
        
        importance[i] = 1 - np.mean(np.round(trained_model.predict(Xhat)) == Y)
        
    return importance