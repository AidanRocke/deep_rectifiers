#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:58:37 2017

@author: aidanrocke
"""

import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras import regularizers
from keras.models import Sequential
#from keras.optimizers import adam
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
import xgboost as xgb
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data
from scipy.stats import pearsonr


#what's in here:
# 1. create_model
# 2. load_model
# 3. feature_selection

def pearson_matrix(X):
    
    N = min(np.shape(X))
    
    corr = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            corr[i][j] = pearsonr(X[:,i],X[:,j])[0]
            
    return corr

def uncertainty(trained_model,validate_X):
    
    N, M = np.shape(validate_X)
    
    predictions = np.zeros((N,20))
    
    uncertainty = np.zeros(N)
    
    indices = np.random.choice(np.arange(M),20,replace = False)
    
    for k in range(20):
        
        i = indices[k]
        
        mu = np.mean(validate_X[:,i])
        sigma = np.std(validate_X[:,i])
        
        Xhat = validate_X
        
        Xhat[:,i] = np.random.normal(mu,sigma)
        
        predictions[:,k] = np.reshape(trained_model.predict(Xhat),(N,))
        
    for j in range(N):
        uncertainty[j] = np.var(predictions[j])
        
    return uncertainty


def experimental_data(competition):
    
    #get data:
    
    file = '/Users/aidanrocke/Desktop/numerai/competition_'+str(competition)+'/'

    train = pd.read_csv(file+'numerai_data/numerai_training_data.csv')


# prepare data:   
    cols = list(train.columns.values)
    
    train_X, train_Y = np.array(train[cols[3:24]]), np.array(train[cols[24:25]])
    #preprocess data:
    
    train_X = preprocess_data(train_X)
    
    train_X = train_X[0]
    
    #apply test train split:
    train_X, validate_X, train_Y, validate_Y = train_test_split(train_X, train_Y, test_size=0.33, random_state=42)
    
    return train_X, validate_X, train_Y, validate_Y


def create_model(layers,dropout,regularization, batch_norm,activation,output,loss,optimizer):
    """
        inputs: 
            
            layers: a list of layers of the neural network // list
            
            dropout: dropout ratio applied to each layer // float
            
            regularization: whether or not to regularize // binary
            
            batch_norm: binary, whether there's dropout or not // int
            
            optimizer: the type of optimizer used // str
            
            loss_func: the loss function used // str
            
        outputs:
            
            model: the keras model that will be used of type Sequential
    
    """
    N = len(layers)
    
    model = Sequential()
    
    
    
    if regularization == 1:
        
        model.add(Dense(units= layers[1],input_dim=layers[0], use_bias=True,kernel_regularizer=regularizers.l2(0.01), kernel_initializer="uniform", activation=activation))

    else: 
        model.add(Dense(units= layers[1],input_dim=layers[0], use_bias=True,kernel_initializer="uniform", activation=activation))
    
    for i in range(1,N-2):
        if regularization == 1:
            model.add(Dense(units= layers[i+1],use_bias=True,kernel_regularizer=regularizers.l2(0.01),kernel_initializer="uniform", activation=activation))
        else: 
            model.add(Dense(units= layers[i+1],use_bias=True,kernel_initializer="uniform", activation=activation))
        
        if batch_norm == 1:
            model.add(BatchNormalization())
        

        #model.add(Dropout(dropout))
        
    model.add(Dense(units=layers[N-1],use_bias=True,kernel_initializer="normal", activation=output))
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
    
    return model


def feature_selection(train_X, train_Y):
    """
    input: 
        train_X : a fully-featured training matrix
    
    output:
        train_X: of type matrix with features removed if these features have
                 below average feature importance
    
    """
    model = xgb.XGBClassifier()
    model.fit(train_X, train_Y.flatten())
    
    importance = model.feature_importances_
    avg = np.mean(importance)
    
    inds = np.array([importance > avg])

    indices = inds[0]
    Z = np.array(range(min(np.shape(train_X))))
    
    #train_X = np.delete(train_X, Z[indices],1)
    
    return Z[indices]

