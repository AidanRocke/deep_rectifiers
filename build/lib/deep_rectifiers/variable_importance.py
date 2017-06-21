#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:26:05 2017

@author: aidanrocke
"""

import numpy as np

def classification_importance(trained_model,validate_X,validate_Y):
    
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

#for regression it might be a better idea to simply backpropagate:

def regression_importance(trained_model,validate_X,validate_Y):
    
    preds = trained_model.predict(validate_X)
    
    X = validate_X
    
    N, M = np.shape(validate_X)
    
    importance = np.zeros(M)
    
    for i in range(M):
        
        mu = np.mean(X[:,i])
        sigma = np.std(X[:,i])
        
        Xhat = X
        
        Xhat[:,i] = np.random.normal(mu,sigma)
        
        new_preds = trained_model.predict(Xhat)
        
        importance[i] = np.linalg.norm(preds-new_preds)
        
    return importance