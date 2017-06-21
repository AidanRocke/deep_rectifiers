#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:27:53 2017

@author: aidanrocke
"""

import numpy as np

from validation import accuracy
from utils import create_model
from sklearn.metrics import log_loss



def bagging(X,Y, n):
    """
        inputs:
            X:
            Y: 
            samples: 
                
        outputs:
            
            a list of noisy training data sampled with replacement
    
    """
    
    samples = []
    
    for i in range(n):
            
        indices = np.array(range(len(Y)))
        sample = np.random.choice(indices,len(indices))
            
        samples.append([X[sample], Y[sample]])
        
    
    return samples

def boosting(models, train_X, train_Y):
    
    indices = np.array(range(len(train_X)))
    sample = np.random.choice(indices,round(len(indices)*0.8), replace = False)
    
    X, Y = train_X[sample], train_Y[sample]
    
    N = len(models)
    
    new_models = []
    
    for i in range(N):
        
        model = models[i]
        
        model.fit(X,Y,batch_size=5000, verbose=1)
        
        new_models.append(model)
        
        probs = np.round(model.predict(train_X))
        
        boolean = np.array([probs != train_Y])
        
        X_errors = train_X[boolean[0][:,0]]
        Y_errors = train_Y[boolean[0][:,0]]
        
        indices = np.array(range(len(X_errors)))
        sample = np.random.choice(indices,round(len(indices)*0.8), replace = False)
        
        
        X, Y = X_errors[sample], Y_errors[sample]
        
    return new_models

def boosting_evaluation(models,X,Y):
    
    N, M = len(Y),len(models)
    
    preds = np.zeros(shape=(N,M))
    yhat = np.zeros(N)
    
    for i in range(M):
        model = models[i]
        probs = model.predict(X)
        preds[:,i] = probs.reshape(N,)
    
    for i in range(N):
        yhat[i] = np.mean(preds[i])
        """
        if np.mean(preds[i] < 0.5) > 0.5:
            inds = np.array(preds[i] < 0.5)
            yhat[i] = np.min(preds[i][inds])
        else:
            inds = np.array(preds[i] > 0.5)
            yhat[i] = np.max(preds[i][inds])
        """
            
            
    return log_loss(Y,yhat)


def stacking(models, train_X, train_Y):
    
    N, M = len(models), len(train_Y)
    
    super_model = create_model([N,500,500,1],0.8,1,'elu','sigmoid','binary_crossentropy')
    
    predictions = np.zeros(shape=(M,N))
    
    new_models = []
    
    for i in range(N):
        model = models[i]
        
        indices = np.array(range(M))
        sample = np.random.choice(indices,round(len(indices)*0.7), replace = False)
    
        X, Y = train_X[sample], train_Y[sample]
        
        model.fit(X,Y,batch_size=5000, verbose=1)
        
        new_models.append(model)
        
        probs = model.predict(train_X)
        
        predictions[:,i] = probs.reshape(M,)
        
    
    #fit the super model to predictions:
        
    super_model.fit(predictions,train_Y)
    
    return super_model, new_models

def stacking_evaluation(models, super_model,X,Y):
    
    N, M = len(Y), len(models)
    
    predictions = np.zeros(shape=(N,M))
    
    for i in range(M):
        model = models[i]
        
        probs = model.predict(X)
        
        predictions[:,i] = probs.reshape(N,)
    
    return accuracy(super_model,predictions,Y)
        
        
    
    