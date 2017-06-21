# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

import sys

sys.path.insert(0,'/Users/aidanrocke/Desktop/scientific_deep_learning/deep_science')
from utils import *

# we assume that the default condition is that the models are used for 
# classification and the integer labels aren't one-hot encoded. 

#get activations:
def get_activity(model, layer, X):
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
    
    return layer_model.predict(X)

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


def binary_activity(models_file_path,num_layers,width,X):
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
    models = [model for model in os.listdir(models_file_path) if model.endswith('.h5')]
        
    N, M = np.shape(X)
    epochs = len(models)
    
    mean_activity = np.zeros((epochs,N,num_layers-1))
    
    activity = np.zeros((epochs,N,width*(num_layers-1)))
    
    

    for i in range(epochs):
  
        model = load_model(models_file_path+models[i])
        
        
        for j in range(num_layers):            
            #get activations:
            activations = np.array(get_activity(model, j, X) > 0,dtype=bool)
            activity[i][:,range(j*width,(j+1)*width)] = activations
            
            layer_values = np.array(activations > 0,dtype=bool)
            
            mean_activity[i][:,j] = percent_active(layer_values.astype(int))
            
                    
    return activity, mean_activity

def subset_variable_size(classes,X_train, X_test, y_train,y_test):
    """
        input: 
            classes (list): a list of integers specifying classes
            
            X_train (numpy.ndarray) : training data
                
            X_test (numpy.ndarray) : test data
                
            y_train (numpy.ndarray): training labels
                
            y_test (numpy.ndarray) : test labels
            
        output: 
            activity (numpy.ndarray) : a multidimensional array representing 
                                       binary activations per sample
    """
    #get conditions by obtaining boolean array:
    c1, c2  = np.array([y_train == i for i in classes])+0, np.array([y_test == i for i in classes])+0 
    
    c1, c2 = np.array([np.max(c1[:,i]) for i in range(len(c1))]), np.array([np.max(c2[:,i]) for i in range(len(c2))])
    
    
    #let's subset our training data:
    X_train, X_test = X_train[c1], X_test[c1]
    
    y_train, y_test = y_train[c2], y_test[c2]
    
    #we create a model:
    model = create_model([784,500,500,500,10],0,0, 0,'relu','softmax','categorical_crossentropy','adam')
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=5000, verbose=2)
    
    N, M = np.shape(X_train)
    
    activity = np.zeros((10,N,1500))
    
    for i in range(3):            
            #get activations:
            activations = np.array(get_activity(model, j, training_data) > 0,dtype=bool)
            activity[i][:,range(j*width,(j+1)*width)] = activations
            
    return activity       

def activation_map(activity,labels,epoch):
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
        
        avg = np.mean(act[np.where(labels == i)],0)
    
        for j in range(10):
        
            delta = act[np.where(labels == j)] - avg
            
            diff[i][j] = np.mean([np.linalg.norm(k) for k in delta])
            
    return diff
    

#apply PCA before clustering:
def visualize_mean_activity(mean_activity):
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
    
    
def variable_size_representation(mean_activity,labels):
    """
        A table ranking 'variable size' using the average number of active nodes for each class. 

        input:
            mean_activity (numpy.ndarray): the fraction of nodes active per sample
    """
        
    global_activity = np.mean(mean_activity[len(mean_activity)-1],1)
    
    N= len(global_activity)
    
    activity = pd.DataFrame(data = np.hstack((global_activity.reshape((N,1)),labels.reshape((N,1)))),columns=['fraction_active', 'label']) 
    
    variable_size = []

    for i in range(10):
        act = activity.loc[activity['label'] == i]
        variable_size.append(act.fraction_active.values)

    average_values = np.array([np.mean(K) for K in variable_size])    
    
    ranks = rankdata(average_values)
    
    rankings = pd.DataFrame(data = ranks.reshape((1,10)), index = ['variable_size'], columns=['rank of '+str(i) for i in range(10)]) #creates a new dataframe that's empty

    return rankings

    
    
    
    

def visualize_activations(activity,labels):
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

    scat = ax.scatter(pca_result[:,0],pca_result[:,1], c=labels,cmap=cm.Set3)
    ax.set_title('PCA activation clusters',fontsize=15)
    ax.set_facecolor('bisque')

    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)


    plt.show()
    
def nodes_set(activity,labels,num_labels):
    
    nodes_used, vectors = [], []
    
    for j in range(num_labels):
    
        vectors.append(activity[np.where(labels == j)])
    
        nodes_used.append(set(np.nonzero(vectors[j])[1]))
        
    return nodes_used
    
    
# node sharing is a more direct measure of competition and cooperation:   
def node_sharing(activity,num_labels, labels):
    
    L, M, N = np.shape(activity)
    
    nodes_shared = np.zeros((L,num_labels,num_labels))
        
    for i in range(L):

        act = activity[i]   
        
        nodes_used = nodes_set(act,labels,num_labels)
            
        for j in range(num_labels):
            
            nodes_shared[i,j] = np.array([len(set.intersection(nodes_used[j],nodes_used[k])) for k in range (num_labels)])/float(len(nodes_used[j]))
            
    return nodes_shared            
    
    #return nodes_shared, sharing_activity

def node_heatmaps(activity,num_labels, labels,nodes_shared,sharing_activity):
    
    L, M, N = np.shape(activity)
    
    nodes_shared, sharing_activity = np.zeros((L,N,N)), np.zeros((L,N,N))
    
    Nodes_Shared, Node_Activity = node_sharing(activity,num_labels, labels,nodes_shared,sharing_activity)
    
    return Nodes_Shared, Node_Activity

# small tests:
"""
nodes_shared = node_sharing(activity,10, y)

sns.heatmap(nodes_shared[0],cmap="Blues")

#if I didn't normalise the matrix, I could 
plt.plot(np.diagonal(nodes_shared[4]))
"""
    
