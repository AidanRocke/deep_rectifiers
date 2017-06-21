#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:25:19 2017

@author: aidanrocke
"""

from sklearn.metrics import log_loss
import numpy as np
import xgboost as xgb

def accuracy(model,X,Y):
    yhat = model.predict(X)
    
    return np.mean(np.round(yhat)==Y), log_loss(Y, yhat)

def cross_validate(model, train_X, train_Y, K,epochs):
    """This function performs K-fold cross-validation"""
    
    #shuffle the training data:
    permutation = np.random.permutation(range(len(train_X)))
    train_X = train_X[permutation]
    train_Y = train_Y[permutation]
    
    
    #generate K random subsamples:
    cols = round(float(len(train_X))/float(K))
    sample_indices = np.reshape(permutation[:K*cols],(K,cols))
    
    #create array for loss and metric(ex. accuracy):
    CV_results = np.zeros((K,2))
    
    #create row index for generating K-samples:
    row_index = np.array(range(K))
    
    for i in range(K):
        I = row_index[np.arange(K) != i]
        rows = np.concatenate(sample_indices[I],axis=0)
                
        #model.fit(train_X[rows], train_Y[rows], epochs= epochs, batch_size=batches, verbose=1)
        model.fit(train_X[rows], train_Y[rows], epochs= epochs, verbose=1)
        
        evaluation = model.evaluate(train_X[sample_indices[i]], train_Y[sample_indices[i]], verbose=1)
        
        CV_results[i] = evaluation[0], evaluation[1]
            
    return CV_results

def adversarial_validation(train_X,test_X):
    
    #prepare data:
    train_Y = np.zeros(len(train_X))
    test_Y = np.ones(len(test_X))
    
    X = np.concatenate((train_X,test_X))
    Y = np.concatenate((train_Y, test_Y))
    
    model = xgb.XGBClassifier()
    model.fit(X, Y.flatten())
    
    probs = model.predict(train_X)
    
    indices = [i[0] for i in sorted(enumerate(probs), key=lambda x:x[1])]
    
    X, Y = train_X[indices], train_Y[indices]
    
    train_X, validate_X = X[1:int(len(indices)*.70)], X[int(len(indices)*.70):len(indices)]
    
    train_Y, validate_Y = Y[1:int(len(indices)*.70)], Y[int(len(indices)*.70):len(indices)]
    
    return train_X, validate_X, train_Y, validate_Y  