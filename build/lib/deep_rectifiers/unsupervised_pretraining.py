#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 23:06:56 2017

@author: aidanrocke
"""

import sys

sys.path.insert(0,'/Users/aidanrocke/Desktop/scientific_deep_learning/deep_science')

from utils import create_model

model = create_model([784,100,10,784],0,0, 0,'relu','linear','binary_crossentropy','adam')

model.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10, batch_size=200, verbose=2)

#get weights:
layers = [model.layers[i] for i in range(3)]

layer_1 = layers[0]

W1 = layer_1.get_weights()

#train a model with these unsupervised weights:
unsup_model = create_model([784,100,100,10,784],0,0, 0,'relu','linear','binary_crossentropy','adam')

unsup_layers = [unsup_model.layers[i] for i in range(3)]

Ulayer_1 = unsup_layers[0]

Ulayer_1.set_weights(W1)

history_1 = unsup_model.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10, batch_size=200, verbose=2)

unsup_layers = [unsup_model.layers[i] for i in range(3)]

Ulayer_1, Ulayer_2 = unsup_layers[0], unsup_layers[1]

U1, U2 = Ulayer_1.get_weights(), Ulayer_2.get_weights()


#try to visualise data:
import matplotlib.pyplot as plt

y_hat = unsup_model.predict(X_train)

image1, image2, image3 = y_hat[0], y_hat[1], y_hat[1000]

Im_1, Im_2, Im_3 = image1.reshape((28,28)), image2.reshape((28,28)), image3.reshape((28,28))

plt.imshow(Im_2)
#Actually displaying the plot if you are not in interactive mode
plt.show()

#check performance of model with unsupervised pretraining:
unsup = create_model([784,100,100,784],0,0, 0,'relu','softmax','binary_crossentropy','adam')

unsup_layers = [unsup.layers[i] for i in range(3)]

Ulayer_1, Ulayer_2 = unsup_layers[0], unsup_layers[1]

Ulayer_1.set_weights(U1)
Ulayer_2.set_weights(U2)

history_2 = unsup.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

unsup.evaluate(X_test, y_test)
# if width = 1000 then [0.015236866146049124, 0.99627999534606937]
# if width = 100 then [0.013412230537246796, 0.99584999771118166]

def get_weights(model):
    K = len(model.layers)
    
    model_layers = [model.layers[i] for i in range(K-1)]
    
    weights = []
    
    for X in model_layers:
        
        weights.append(X.get_weights())
        
    return weights

def change_weights(model_1,model_2):
    
    weights = get_weights(model_1)
    
    mod2_layers = [model_2.layers[i] for i in range(len(weights))]
    
    i = 0
    for X in mod2_layers:
        X.set_weights(weights[i])
        i +=1
        
    return model_2

sup = create_model([784,100,100,10],0,0, 0,'relu','softmax','binary_crossentropy','adam')

def semisupervised_training(unsup,layers,data,epochs):
    
    X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
    
    sup = create_model(layers,0,0, 0,'relu','softmax','binary_crossentropy','adam')
    
    for i in range(3):
        sup.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)
        
        unsup = change_weights(sup,unsup)
        
        unsup.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=1, batch_size=200, verbose=2)
        
        sup = change_weights(unsup,sup)
        
    for j in range(epochs):
        sup.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)


    return sup

data = [X_train, y_train, X_test, y_test]

sup = semisupervised_training(unsup,[784,100,100,10],data,10)
    
sup.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

        

weights = get_weights(unsup)
    
    

    
