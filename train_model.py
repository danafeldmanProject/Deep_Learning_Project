# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:36:17 2021

@author: danaf
"""


import os
import cv2
import numpy as np

import prints



    
def Put_Into_Lists(train_directory_path,test_directory_path,class_names,image_size): 
    '''
        puts the data and labels into arrays
        resizes images
        returns the arrays
    '''
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    print(class_names_label)
    nb_classes = len(class_names)
    
    datasets = [train_directory_path,test_directory_path]
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for img in os.listdir(os.path.join(dataset, folder)):
                
                # Get the path name of the image
                img_path = os.path.join((os.path.join(dataset, folder)),img)
                ###print(img_path)
                
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.resize(image, image_size) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))
    return output
        
        
        
    

def Train(model,train_images, train_labels):
    '''
        train the model with train data and labels
    '''
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(train_images, train_labels, batch_size=64, epochs=7, validation_split = 0.4)
    model.save('model.h5')
    
    return history




def test_model(model,test_images, test_labels):
    '''
        test the model with test data and labels
    '''
    test_loss = model.evaluate(test_images, test_labels)    
    predictions = model.predict(test_images)     # Vector of probabilities
    pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
    

def handle_train(model,train_images, train_labels):
    """
    manage the train section
    return history
    """
    history = Train(model,train_images, train_labels)
    prints.plot_accuracy_loss(history)
    return history

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
