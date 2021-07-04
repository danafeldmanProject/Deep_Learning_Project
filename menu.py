# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:04:17 2021
@author: danaf
"""

import os 
import seaborn as sn; sn.set(font_scale=1.4)
import tensorflow as tf 
from sklearn.utils import shuffle

import datamod
import my_Model
import train_model
import prints


#DIRECTORY_PATH=r'C:\Users\danaf\OneDrive\Documents' #main directory location 
#ORIGINAL_PATH=r'C:\Users\danaf\OneDrive\Documents'   #where the zip was unloaded
#IMAGE_SIZE = (150, 150) 
#CLASSES=["men","women"]
#MODEL_PATH=r'C:\Users\danaf\OneDrive\Desktop\DeepLearning\project\classification\model.h5' 


DIRECTORY_PATH=r'C:\Users\Student\OneDrive\Documents\Deep_Learning_Project-main' #main directory location 
ORIGINAL_PATH=r'C:\Users\Student\OneDrive\Documents'   #where the zip was unloaded
IMAGE_SIZE = (150, 150) 
CLASSES=["men","women"]
MODEL_PATH=r'C:\Users\Student\OneDrive\Documents\Deep_Learning_Project-main\model.h5'

    
def handle_data(plot_graphs):    
    '''
    make arrays of data (images and labels), return them
    normelize pixel values
    if plot_graphs==True:
        plot the information about the data
    '''
    path=DIRECTORY_PATH
    main_directory_path=os.path.join(path,"dataset")
    class_names=CLASSES
    img_size=IMAGE_SIZE
    (train_images, train_labels), (test_images, test_labels) = train_model.Put_Into_Lists(os.path.join(main_directory_path, "train"),
                                                                                          os.path.join(main_directory_path, "test"),
                                                                                          class_names,
                                                                                          img_size)
    train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
    
    #normelize pixel values
    train_images = train_images / 255.0 
    test_images = test_images / 255.0
    
    n_train = train_labels.shape[0]
    n_test = test_labels.shape[0]
    
    if plot_graphs:
        prints.Plot_Information(train_labels,test_labels,n_train,n_test,class_names,img_size)
    
    return (train_images, train_labels), (test_images, test_labels)


def check_path(path):
    '''
    check whether path is valid (exists)
    '''
    isFile = os.path.isdir(path)
    if isFile==False:
        prints.printError("path invalid, enter again")
        return False
    else:
        return True
    
    
def path_inputs():
    '''
    get inputs for paths
    '''
    #path for saved model
    done=False
    while done==False:
        prints.printOptions("enter project path- where you saved the .py files: ")
        project_path=input()
        done=check_path(project_path)    
    MODEL_PATH=os.path.join(project_path,"model.h5")
    
    done=False
    while done==False:
        prints.printOptions("enter zip path: ")
        zip_file_path=input()
        done=check_path(project_path)
    
    done=False
    while done==False:
        prints.printOptions("enter directory to unload dataset into: ")
        DIRECTORY_PATH=input()
        done=check_path(project_path)
    
    #extract dataset 
    path=DIRECTORY_PATH
    datamod.extract_Zip(zip_file_path,path) 
    
  
    
    
    
def case_one():
    '''
    get the data ready for training\testing\predicting
    '''
    #get the paths
    path_inputs()

    #arrange data into directories and transform to tensors       
    (train_images, train_labels), (test_images, test_labels)=handle_data(plot_graphs=True)
    
    return (train_images, train_labels), (test_images, test_labels)


def case_two():
    '''
    train model
    '''
    (train_images, train_labels), (test_images, test_labels)=handle_data(plot_graphs=False)
    model=my_Model.build_model()
    history=train_model.handle_train(model,train_images, train_labels)
    

    
def case_three():
    '''
    test model
    '''
    (train_images, train_labels),(test_images, test_labels)=handle_data(plot_graphs=False)
    trained_model_path=MODEL_PATH
    train_model.test_model(tf.keras.models.load_model(trained_model_path),test_images, test_labels)


def options():
    """
    This function prints for the user his options (UI)
    """
    prints.printOptions("******************************************")
    prints.printOptions("*          USER INTERFACE                *")
    prints.printOptions("*                                        *")
    prints.printOptions("*   Enter 1 --> get data information     *")
    prints.printOptions("*   Enter 2 --> train the model          *")
    prints.printOptions("*   Enter 3 --> test the model           *")
    prints.printOptions("*   Enter space bar --> exit             *")
    prints.printOptions("*                                        *")
    prints.printOptions("******************************************")

    
def Main():
    
    options()
    flag=True
    while(flag):
        prints.printOptions("--> Your Choice: ")
        choice = input("Enter: ")
        
        if choice == '1':
            """
            if the use enter 1 -> the directory of the data will be updated
            """
            (train_images, train_labels),(test_images, test_labels)=case_one()
            prints.printProcess("[INFO] Using new data set")
    
        if choice == '2':
            """
            if the use enter 2 -> the directory of the model and the labeld will be updated
            """
            case_two()
            prints.printProcess("[INFO] Using trained model")
         
        if choice == '3':
            """
            if the use enter 3 -> the program will use the updated directory and test the model
            """   
            case_three()

            
        if choice == ' ':
            prints.printProcess("[INFO] Exiting...")
            flag = False
    
    
    
    
    
    
    
Main()
    
    
    
    
    
    
    
     
    
    
    
    
