#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:28:05 2017

@author: aidanrocke
"""

import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from scipy.stats import entropy
from keras.models import load_model
import os
import pandas as pd

from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.cm as cm

"""
import sys

sys.path.insert(0,'/Users/aidanrocke/Desktop/deep_rectifiers/deep_rectifiers/')
from utils import *
"""

class Activations:
    def __init__(self,models_file_path,num_layers,width,Data):
        self.file_path = models_file_path
        self.num_layers = num_layers
        self.width = width
        self.X_train, self.X_test = Data[0], Data[1]
        self.y_train, self.y_test = Data[2], Data[3]
        self.num_labels = len(set(self.y_train))

    def get_activity(self,model, layer):
        """
            input: 
                model (keras.models.Sequential) : a Keras sequential model 
                layer (int): a particular hidden layer in a model 
                X (numpy.ndarray): samples that can be fed to a model 
            
            output: 
                activity (numpy.ndarray) : activations of a particular hidden layer
        
        """
        layer = model.layers[layer]
        layer_model = Model(inputs=model.input,outputs=layer.output)
        
        return layer_model.predict(self.X)

    def percent_active(activations):
        """
            input:
                activations (numpy.ndarray): a multidimensional array of binary
                                             activations
                                             
            output:
                fraction (numpy.ndarray) : the fraction of nodes active per sample
        """
        N, M = np.shape(activations)
        
        fraction = np.zeros(N)
        
        for i in range(N):
            fraction[i] = np.mean(activations[i])
            
        return fraction


    def binary_activity(self):
        """
            input:
                models_file_path (str) : location of folder containing trained Keras
                                         models where each model is assumed to have
                                         the same architecture
                
                num_layers (int) : the number of layers of each model
                
                width (int) : the width of the hidden layers of each model
                
                X (numpy.ndarray) : samples that can be fed to a model
                
            output: 
                activity (numpy.ndarray) : a multidimensional array representing 
                                           binary activations per sample
                
                mean_activity (numpy.ndarray) : the fraction of nodes active per sample
        """
        
        #load models:
        models = [model for model in os.listdir(self.file_path) if model.endswith('.h5')]
            
        N, M = np.shape(self.X)
        epochs = len(models)
        
        mean_activity = np.zeros((epochs,N,self.num_layers))
        
        activity = np.zeros((epochs,N,self.width*(self.num_layers)))
        
        

        for i in range(epochs):
      
            model = load_model(self.file_path+models[i])
            
            
            for j in range(self.num_layers):            
                #get activations:
                activations = np.array(Activations.get_activity(model, j, self.X) > 0,dtype=bool)
                activity[i][:,range(j*self.width,(j+1)*self.width)] = activations
                
                layer_values = np.array(activations > 0,dtype=bool)
                
                mean_activity[i][:,j] = Activations.percent_active(layer_values.astype(int))
                
                        
        return activity, mean_activity
    
    
    def variable_size_representation(self,classes,model):
        """
            input: 
                classes (list): a list of integers specifying classes
                
                model (Sequential) : a particular model to use
                
            output: 
                rankings (Dataframe) : a multidimensional array representing 
                                           binary activations per sample
        """
        #get conditions by obtaining boolean array:
        ix1, ix2  = np.array([self.y_train == i for i in classes])+0, np.array([self.y_test == i for i in classes])+0 
        
        ix1, ix2 = np.array([np.max(ix1[:,i]) for i in range(len(ix1))]), np.array([np.max(ix2[:,i]) for i in range(len(ix2))])
        
        
        #let's subset our training data:
        X_train, y_train = self.X_train[ix1], self.y_train[ix1]
        
        X_test, y_test = self.X_train[ix2], self.y_test[ix2]
        
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=5000, verbose=2)
        
        
        #get activity per layer:
        N, M = np.shape(X_train)
        
        activity_per_layer = np.zeros((N,self.num_layers))
        
        for i in range(self.num_layers):            
                #get activations:
                activations = np.array(Activations.get_activity(model, i, X_train) > 0,dtype=bool)
                
                layer_values = np.array(activations > 0,dtype=bool)
                
                activity_per_layer[:,i] = Activations.percent_active(layer_values.astype(int))
                
        #rank the variable sizes:
        mean_activity = np.mean(activity_per_layer,1)
        
        activity = pd.DataFrame(data = np.hstack((mean_activity.reshape((N,1)),y_train.reshape((N,1)))),columns=['fraction_active', 'label']) 
        
        variable_size = []

        for i in classes:
            act = activity.loc[activity['label'] == i]
            variable_size.append(act.fraction_active.values)
    
        average_values = np.array([np.mean(K) for K in variable_size])    
        
        ranks = rankdata(average_values)
        
        rankings = pd.DataFrame(data = ranks.reshape((1,len(classes))), index = ['variable_size'], columns=['rank of '+str(i) for i in range(10)]) #creates a new dataframe that's empty
    
        return rankings
    
    def visualize_activations(self,activity):
        """
            Two dimensional linear embedding of binary activity. 
        """
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(activity)
    
    
        plt.style.use('ggplot')
    
        bounds = np.linspace(0,10,11)
    
        #create plots:
    
        f, (ax) = plt.subplots(1, 1,figsize=(25,15))
    
        #, sharey=True
    
        scat = ax.scatter(pca_result[:,0],pca_result[:,1], c=self.y_train,cmap=cm.Set3)
        ax.set_title('PCA activation clusters',fontsize=15)
        ax.set_facecolor('bisque')
    
        plt.colorbar(scat, spacing='proportional',ticks=bounds)
    
    
        plt.show()
        
    def visualize_average_sparsity(mean_activity):
        """
            Visualization of sparsity of each hidden layer's activity using histograms 
            for each epoch. 
        """
        
        epochs, samples, layers = np.shape(mean_activity)
        
    
        f, ax = plt.subplots(layers, epochs,figsize=(20,20))
        
        plt.style.use('ggplot')
    
    
        for i in range(epochs):
            for j in range(layers):
                
                ent = entropy(mean_activity[i][:,j],mean_activity[epochs-1][:,j])
                ax[j,i].hist(mean_activity[i][:,j],color='steelblue',label='entropy = '+str(ent))
                ax[j,i].set_title('epoch '+str(i+1)+'_layer '+str(j+1))
                
                ax[j,i].legend(loc='upper left')
    
        plt.show()
        
    def activation_map(self,activity,epoch):
        """
    
            input: 
                activity (numpy.ndarray) : a multidimensional array representing 
                                           binary activations per sample
    
                labels (numpy.ndarray) : the target label for each sample 
    
                 epoch (int) : a version of a model at an epoch in its training history
    
            output: 
    
                diff (numpy.ndarray): a map of the average euclidean distance in activation space
        
        """
    
        act = activity[epoch]
        
        diff = np.zeros((10,10))
        
        for i in range(10):
            
            avg = np.mean(act[np.where(self.X_train == i)],0)
        
            for j in range(10):
            
                delta = act[np.where(self.y_train == j)] - avg
                
                diff[i][j] = np.mean([np.linalg.norm(k) for k in delta])
                
        return diff
    
    #analysing of node sharing among subnetworks:
    
    def nodes_set(self,activity,num_labels):
    
        nodes_used, vectors = [], []
                
        for j in range(self.num_labels):
        
            vectors.append(activity[np.where(self.y_train == j)])
        
            nodes_used.append(set(np.nonzero(vectors[j])[1]))
            
        return nodes_used
    
    def node_sharing(self,activity,num_labels):
    
        L, M, N = np.shape(activity)
        
        nodes_shared = np.zeros((L,num_labels,num_labels))
            
        for i in range(L):
    
            act = activity[i]   
            
            nodes_used = Activations.nodes_set(act,self.y_train,self.num_labels)
                
            for j in range(self.num_labels):
                
                nodes_shared[i,j] = np.array([len(set.intersection(nodes_used[j],nodes_used[k])) for k in range (num_labels)])/float(len(nodes_used[j]))
                
        return nodes_shared     
    
    
    
    
    
    