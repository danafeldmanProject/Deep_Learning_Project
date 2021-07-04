# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:21:35 2021

@author: danaf
"""

import seaborn as sn; sn.set(font_scale=1.4)
import keras
from keras.layers.core import Dropout


def build_model():           

    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(32,(5,5),padding='same',activation='relu',input_shape=(150,150,3)))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(Dropout(0.2))

    
    # Convolutional layer and maxpool layer 2
    model.add(keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(Dropout(0.2))

    
    # Convolutional layer and maxpool layer 3
    model.add(keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(Dropout(0.2))

    
    # Convolutional layer and maxpool layer 4
    model.add(keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(Dropout(0.2))

    
    # This layer flattens the resulting image array to 1D array
    model.add(keras.layers.Flatten())
    
    # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(Dropout(0.2))

    
    # Output layer with single neuron which gives 0 for Female or 1 for Male 
    #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    
    
    return model
