#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 03:43:34 2017

@author: aidanrocke
"""
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from keras.models import load_model
import numpy as np
import os 
from keras.models import load_model  

filepath = '/Users/aidanrocke/Desktop/scientific_deep_learning/deep_science/experiments/models/three_layer/'

model = load_model(filepath+'3.h5')

layers = [model.layers[i] for i in range(3)]

layer_1, layer_2, layer_3 = layers[0], layers[1], layers[2]

W1 = layer_1.get_weights()[0]

W2 = layer_2.get_weights()[0]

W3 = layer_3.get_weights()[0]

def ortho(matrix):
        
    return np.mean(np.cov(matrix))

#the matrices are clearly orthonormal:
score1, score2, score3 = ortho(W1), ortho(W2), ortho(W3)

def ortho_gauss(samples):
    
    random = np.zeros(samples)
    
    for i in range(samples):

        Z = np.zeros((500,500))
    
        for j in range(500):
            Z[j] = np.random.normal(size = 500)
        
        random[i] = ortho(Z)
        
    return np.mean(random), np.var(random), random


x,y,z = ortho_gauss(10)

#now let's analyse whether there is convergence:
    
def analyse_convergence(file_path,num_models):


    models = []
    
    for i in range(num_models):
        models.append(load_model(filepath+str(i)+'.h5'))
    
    for i in range(num_models):
        
        model = models[i]
        
        num_layers = len(model.layers)-1
        
        scores = np.zeros((num_models,num_layers))
        
        layers = [model.layers[i] for i in range(num_layers)]
        
        weights = []
        
        for j in range(num_layers):
        
            layer = layers[j]
            
            W_matrix = layer.get_weights()[0]
            
            #initialise_control matrix:
            Z = np.zeros((500,500))
        
            for k in range(500):
                Z[k] = np.random.normal(loc = np.mean(500*W_matrix), scale = np.var(500*W_matrix), size = 500)
            
            weights.append(float(ortho(500*W_matrix))/float(ortho(Z)))
        
        scores[i] = np.array(weights)
    
    
    return scores

def get_weights(model):
    K = len(model.layers)
    
    model_layers = [model.layers[i] for i in range(K-1)]
    
    weights = []
    
    for X in model_layers:
        
        weights.append(X.get_weights())
        
    return weights

def weight_norms(model_path):
    # check weights:
    #model_path = '/Users/aidanrocke/Desktop/scientific_deep_learning/deep_science/experiments/models/three_layer/'

    model_files = os.listdir(path)
    
    N = len(files)

    weight_norms = np.zeros(N)
    models = []

    for i in range(N):
        models.append(load_model(model_path+model_files[i]))
    
    #get weights for each model:
    for i in range(N):
        model = models[i]
        
        W = get_weights(model)
    
        weights[i] = np.mean([np.linalg.norm(x[0]) for x in W])
        
    return weights

def cov_ratio(matrix):

    Z = np.zeros((500,500))
            
    for k in range(500):
        Z[k] = np.random.normal(loc = np.mean(500*matrix), scale = np.var(500*matrix), size = 500)
    
    return float(ortho(500*matrix))/float(ortho(Z))
    
