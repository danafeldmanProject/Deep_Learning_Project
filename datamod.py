# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:58:54 2021

@author: danaf
"""

'''
this module deals with the data
'''
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from zipfile import ZipFile

import prints




def make_directory(parent_dir,new_directory_name):
    '''
    the function gets a directory name and directory path from the user and creates a directory according to that input,
    if the directory doesn't already exist.
    if the user presses enter when inputing directory path, default is desktop
    the function creates sub-direcories named "train" and "test" in the input directory
    '''
    #make sub-directory
    path1 = os.path.join(parent_dir,new_directory_name)

    #create directory if doesn't exist
    isFile = os.path.isdir(path1)   
    if isFile is False:  
        os.mkdir(path1) 
        print(path1+" directory made") 
    else:
        print(path1+" directory already exists")

    return path1



def arrange_directories(parent_directory,directory_name,class_names):
    '''
    the function gets a directory name and directory path from the user
    the fuction creates the directories for the arrangment of the dataset:
        parent directory
        two sub-directories for train and test
        for both train and test: two sub directories for the classes
    '''
    #make sub-directory
    directory_path = make_directory(parent_directory,directory_name)
    make_directory(directory_path,"train")
    make_directory(directory_path,"test")
    for class_name in class_names:
        for directory in os.listdir(directory_path):
             make_directory(os.path.join(directory_path,directory), class_name)
    print("done")
    return directory_path
      


       
def arrange_dataset(original_path,destination_main_path):
    '''
        arrange the dataset into directories
    '''
    train_moved=False
    moved=False
    while moved==False:
        for directory in os.listdir(destination_main_path):
            if directory=="train" and train_moved==False:
                for class_name in os.listdir(os.path.join(destination_main_path,directory)):
                    dir_path=os.path.join(os.path.join(destination_main_path,directory),class_name)
                    pic_path=os.path.join(original_path,class_name)
                    move_pictures(dir_path,pic_path,True)
                    #print("moved")
                train_moved=True
    
            if directory=="test" and train_moved:  
                for class_name in os.listdir(os.path.join(destination_main_path,directory)):
                    dir_path=os.path.join(os.path.join(destination_main_path,directory),class_name)
                    pic_path=os.path.join(original_path,class_name)
                    move_pictures(dir_path,pic_path,False)
                    #print("moved2")
                moved=True



           

def move_pictures(dir_path,pic_path,isTrain):
    '''
    inputs:destination directory to move files to, directory to move files from, boolian for if train or not
    
    the function moves files to directories:
        -if the destination directory is train, move 70% of files from pic_path to train
        -if the destination directory is test, move all of the files from pic_path to test
    '''

    files = os.listdir(pic_path)  #file list from pic_path
    if isTrain: #move 70% of files
        print(isTrain)
        i=(int)(len(files)*0.8)
        print(i)
        for file in files:
            if(i>0):
                shutil.move(os.path.join(pic_path, file), dir_path)
                i-=1  
    else:#move all files
        for file in files:
            shutil.move(os.path.join(pic_path, file), dir_path)




def find_gif(pic_path):
    '''
        check if a path is .gif file, delete if so
    '''
    for class_name in os.listdir(pic_path):
        sub_path = os.path.join(pic_path, class_name)
        for img in os.listdir(sub_path):
            if img.endswith('.gif'):
                print(img)
                os.remove(os.path.join(sub_path, img))               

                
def Print_plot_pics(path):
    '''
    input:path of directory
    the function prints all images in a directory and their names
    '''

    files = os.listdir(path) 
    for file in files:
        print(file)#image name
        plt.imshow(cv2.imread(os.path.join(path,file)))
        plt.show()
   


def extract_Zip(zip_path,parent): 
    
    path = zip_path
    """
    we cheak if the file/folder is zip by its ending - file type
    """
    if path.endswith('.zip'):
        path = parent 
        """
        extract the folder to new location that the user chose
        """        
        with ZipFile(zip_path, 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall(path)
            prints.printProcess("extracted")
        
    else:
        prints.printProcess("not zip file")
    return path


    


    
    
     


    

