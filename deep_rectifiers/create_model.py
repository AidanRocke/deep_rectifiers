#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:58:37 2017

@author: aidanrocke
"""

from keras.layers.core import Dense, Dropout
from keras import regularizers
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

def create_model(layers,dropout,regularization, batch_norm,activation,output,loss,optimizer):
    """
        inputs: 
            
            layers: a list of layers of the neural network // list
            
            dropout: dropout ratio applied to each layer // float
            
            batch_norm: binary, whether there's dropout or not // int
            
            optimizer: the type of optimizer used // str
            
            loss_func: the loss function used // str
            
        outputs:
            
            model: the keras model that will be used of type Sequential
    
    """
    N = len(layers)
    
    model = Sequential()

    model.add(Dense(units= layers[1],input_dim=layers[0], use_bias=True,kernel_initializer="uniform", activation=activation))
    
    for i in range(1,N-2):
        
        model.add(Dense(units= layers[i+1],use_bias=True,kernel_initializer="uniform", activation=activation))
        
        if batch_norm == 1:
            model.add(BatchNormalization())
        

        #model.add(Dropout(dropout))
        
    model.add(Dense(units=layers[N-1],use_bias=True,kernel_initializer="normal", activation=output))
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
    
    return model
