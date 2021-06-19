# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:57:20 2021

@author: danaf
"""


"""
    this file contains the function that prints messages for the user in colors
"""

from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def printOptions(message):
    """
    Get: message
    prints it in green
    """
    init(convert=True)
    print(Fore.GREEN + message) 
    Style.RESET_ALL
    
    
def printError(message):
    """
    Get: message
    prints it in red
    """
    init(convert=True)
    print(Fore.RED + message) 
    Style.RESET_ALL
    
    
def printProcess(message):
    """
    Get: message
    prints it in blue
    """
    init(convert=True)
    print(Fore.BLUE + message) 
    Style.RESET_ALL
    
    
def Plot_Information(train_labels,test_labels,n_train,n_test,CLASSES,IMAGE_SIZE):
    print ("Number of training examples: {}".format(n_train))
    print ("Number of testing examples: {}".format(n_test))
    print ("Each image is of size: {}".format(IMAGE_SIZE))
    
    _, train_counts = np.unique(train_labels, return_counts=True)
    _, test_counts = np.unique(test_labels, return_counts=True)
    pd.DataFrame({'train': train_counts,
                  'test': test_counts}, 
                 index=CLASSES
                ).plot.bar()
    plt.show()
    
    
    plt.pie(train_counts,
        explode=(0, 0) , 
        labels=CLASSES,
        autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Proportion of each observed category')
    plt.show()
    
    
    
def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['acc'],'bo--', label = "acc")
    plt.plot(history.history['val_acc'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()   