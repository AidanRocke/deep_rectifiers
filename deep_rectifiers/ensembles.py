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
import os


class Ensemble:
    def __init__(self,models_file_path,Data):
        self.file_path = models_file_path
        self.models = [model for model in os.listdir(self.file_path) if model.endswith('.h5')]
        self.X_train, self.X_test = Data[0], Data[1]
        self.y_train, self.y_test = Data[2], Data[3]


    def bagging(self, n):
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
                
            indices = np.array(range(len(self.y_train)))
            sample = np.random.choice(indices,len(indices))
                
            samples.append([self.X_train[sample], self.y_train[sample]])
            
        
        return samples

    def boosting(self):
        
        indices = np.array(range(len(self.X_train)))
        sample = np.random.choice(indices,round(len(indices)*0.8), replace = False)
        
        X, Y = self.X_train[sample], self.y_train[sample]
        
        N = len(self.models)
        
        new_models = []
        
        for i in range(N):
            
            model = self.models[i]
            
            model.fit(X,Y,batch_size=5000, verbose=1)
            
            new_models.append(model)
            
            probs = np.round(model.predict(self.X_train))
            
            boolean = np.array([probs != self.y_train])
            
            X_errors = self.X_train[boolean[0][:,0]]
            Y_errors = self.y_train[boolean[0][:,0]]
            
            indices = np.array(range(len(X_errors)))
            sample = np.random.choice(indices,round(len(indices)*0.8), replace = False)
            
            
            X, Y = X_errors[sample], Y_errors[sample]
            
        return new_models



    def stacking(self):
        
        N, M = len(self.models), len(self.y_train)
        
        super_model = create_model([N,500,500,1],0.8,1,'elu','sigmoid','binary_crossentropy')
        
        predictions = np.zeros(shape=(M,N))
        
        new_models = []
        
        for i in range(N):
            model = self.models[i]
            
            indices = np.array(range(M))
            sample = np.random.choice(indices,round(len(indices)*0.7), replace = False)
        
            X, Y = self.X_train[sample], self.y_train[sample]
            
            model.fit(X,Y,batch_size=5000, verbose=1)
            
            new_models.append(model)
            
            probs = model.predict(self.X_train)
            
            predictions[:,i] = probs.reshape(M,)
            
        
        #fit the super model to predictions:
            
        super_model.fit(predictions,self.y_train)
        
        return super_model, new_models