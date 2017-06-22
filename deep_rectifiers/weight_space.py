#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 03:43:34 2017

@author: aidanrocke
"""

import numpy as np
import os 


class Weights:
    def __init__(self,models_file_path,num_layers):
        self.file_path = models_file_path
        self.num_layers = num_layers
        self.models = [model for model in os.listdir(self.file_path) if model.endswith('.h5')]

    def ortho(matrix):
            
        return np.mean(np.cov(matrix))


    def cov_ratio(matrix):
    
        Z = np.zeros((500,500))
                
        for k in range(500):
            Z[k] = np.random.normal(loc = np.mean(500*matrix), scale = np.var(500*matrix), size = 500)
        
        return float(Weights.ortho(500*matrix))/float(Weights.ortho(Z))
    

#the matrices are clearly orthonormal:

    def ortho_gauss(samples):
        
        random = np.zeros(samples)
        
        for i in range(samples):
    
            Z = np.zeros((500,500))
        
            for j in range(500):
                Z[j] = np.random.normal(size = 500)
            
            random[i] = Weights.ortho(Z)
            
        return np.mean(random), np.var(random), random

#now let's analyse whether there is convergence:
    
def analyse_convergence(self):

    num_models = len(self.models)
    
    for i in range(num_models):
        
        model = self.models[i]
                
        scores = np.zeros((num_models,self.num_layers))
        
        layers = [model.layers[i] for i in range(self.num_layers)]
        
        weights = []
        
        for j in range(self.num_layers):
        
            layer = layers[j]
            
            W_matrix = layer.get_weights()[0]
            
            #initialise_control matrix:
            Z = np.zeros((500,500))
        
            for k in range(500):
                Z[k] = np.random.normal(loc = np.mean(500*W_matrix), scale = np.var(500*W_matrix), size = 500)
            
            weights.append(float(Weights.ortho(500*W_matrix))/float(Weights.ortho(Z)))
        
        scores[i] = np.array(weights)
    
    
    return scores

    def get_weights(model):
        K = len(model.layers)
        
        model_layers = [model.layers[i] for i in range(K-1)]
        
        weights = []
        
        for X in model_layers:
            
            weights.append(X.get_weights())
            
        return weights

    def weight_norms(self):
        
        N = len(self.models)
        
        #get weights for each model:
        for i in range(N):
            model = self.models[i]
            
            W = get_weights(model)
        
            weights[i] = np.mean([np.linalg.norm(x[0]) for x in W])
            
        return weights


