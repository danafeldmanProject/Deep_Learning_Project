# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:21:35 2021

@author: danaf
"""

import seaborn as sn; sn.set(font_scale=1.4)
import tensorflow as tf 
                


def build_model():           
    ###make it keras.sewuential
    
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 
    tf.keras.layers.MaxPooling2D(2,2),    
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),   
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
   
    
    #model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.summery()
    
    return model