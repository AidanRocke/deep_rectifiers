#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:50:13 2017

@author: aidanrocke
"""

from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils

from scipy.stats import mode
import numpy as np

def ranking_stats(rankings):
    
    #get the mean, mode and variance
    
    stats = np.zeros((3,len(rankings)))
    
    for i in range(len(rankings)):
        stats[:,i][0] = mode(rankings[:,i])
        stats[:,i][1] = np.mean(rankings[:,i])
        stats[:,i][2] = np.var(rankings[:,i])
        
    return stats

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
        
        if dropout != 0:
            model.add(Dropout(dropout))
        
    model.add(Dense(units=layers[N-1],use_bias=True,kernel_initializer="normal", activation=output))
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
    
    return model

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    train_labels = y_train
    test_labels = y_test
    
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    Data = [X_train,X_test,y_train,y_test]
    
    labels = [train_labels,test_labels]
    
    return Data, labels
        
def train_model(model,data,epoch,path):
    
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1000, verbose=2)
    
    model.save(path+str(epoch)+'.h5')