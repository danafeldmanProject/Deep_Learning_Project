# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:21:35 2021

@author: danaf
"""

import seaborn as sn; sn.set(font_scale=1.4)
import tensorflow as tf 
import keras
                


def build_model():           
    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(keras.layers.MaxPool2D(2,2))
    
    # Convolutional layer and maxpool layer 2
    model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))
    
    # Convolutional layer and maxpool layer 3
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))
    
    # Convolutional layer and maxpool layer 4
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))
    
    # This layer flattens the resulting image array to 1D array
    model.add(keras.layers.Flatten())
    
    # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
    model.add(keras.layers.Dense(512,activation='relu'))
    
    # Output layer with single neuron which gives 0 for Cat or 1 for Dog 
    #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    
    #model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.summery()
    
    return model
