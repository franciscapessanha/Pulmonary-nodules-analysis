#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:42:19 2018

@author: mariafranciscapessanha
"""
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

"""
    Masks e Imagens - fatia do meio (todos os sets)
    
"""
def getData(mode = "default"):
   nodules, masks, metadata = loadData()
   x_train, y_train, masks_train, x_test, y_test, masks_test, x_val, y_val, masks_val = getSets(nodules, metadata, masks, mode) 
   train_slices, train_slices_masks, test_slices, test_slices_masks, val_slices, val_slices_masks = getImages(x_train, masks_train, x_test, masks_test, x_val, masks_val, mode )
   
   return train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val
   

"""
Find Extensions
===============
Finds the files on the directory with the extension provided.

Arguments:
    * directory: path in which we want to find a file
    * extension: type of file. (.npy by default)
    
Returns:
    * files: list of all the files found
    * full path: full path to each of the files
"""
    
def findExtension(directory,extension='.npy'):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
            full_path += [os.path.join(directory,file)]
            
    files.sort()
    full_path.sort()
    return files, full_path

"""
Load Data
================
Returns the middle slice of an volume (in this case a cube)

Returns:
    * nodules: numpy array with the the nodules names and paths
    * masks: numpy array the the masks names and paths
    * metadata: ground truth
    
"""

def loadData():

    #find the current working directory
    curr_path = os.getcwd()
    
    #find the files
    nodule_names, nodules = findExtension(os.path.join(curr_path,'images'))
    #remove the extension from the nodule names
    nodule_names = [os.path.splitext(x)[0] for x in nodule_names]
    nodules = np.asarray([nodule_names, nodules])
    
    mask_names, masks = findExtension(os.path.join(curr_path,'masks'))
    #remove the extension from the mask names
    mask_names = [os.path.splitext(x)[0] for x in mask_names]
    masks = np.asarray([mask_names, masks])
    
    #read the metadata
    metadata = pd.read_excel('ground_truth.xls')
    
    return nodules, masks, metadata


"""
Split Data
===============================
Split the data in:
    * training set (70%)
    * validation set (15%)
    * test set (15%)
    
Arguments:
    * nodules:numpy array with the the nodules names and paths
    * labels: numpy array with the texture label
    
Returns:
    * train: list with the nodules and labels for the train set
    * test: list with the nodules and labels for the test set
    * val: list with the nodules and labels for the validation set
"""

def splitData(data, labels):

    x_train, xi_test, y_train, yi_test = train_test_split(data, labels, test_size = 0.3, random_state=0)
    xf_test, x_val, yf_test, y_val = train_test_split(xi_test, yi_test, test_size = 0.5, random_state=0)
    
    train = [x_train, y_train]
    test = [xf_test, yf_test]
    val =[x_val, y_val]
   
    return train, test, val

def splitTextures(nodules, metadata):
    non_solid = []
    sub_solid = []
    solid = []
    for index in range(nodules.shape[1]):
        texture = int(metadata[metadata['Filename']==nodules[0, index]]['texture'])
    
        if texture <=2:
            non_solid.append(index)
        elif texture >= 3 and texture <= 4:
            sub_solid.append(index)
        elif texture == 5:
            solid.append(index)
    
    non_solid = np.asarray(non_solid)
    sub_solid = np.asarray(sub_solid)
    solid = np.asarray(solid)
    
    ns_nodules = [nodules[:,i] for i in non_solid]
    ss_nodules = [nodules[:,i] for i in sub_solid]
    s_nodules = [nodules[:,i] for i in solid]

    return ns_nodules, ss_nodules, s_nodules

def assignMasks(set_, masks):
    new_masks = []
    for i in range(len(set_)):
        
        for j in range(masks.shape[1]):
            if masks[0,j] == set_[i,0]:
                
                new_masks.append(masks[:,j])

    return np.asarray(new_masks)

"""
labels:
    solid = 2
    sub_solid = 1
    non_solid = 0
"""

def getSets(nodules, metadata, masks, mode):
    
    ns_nodules, ss_nodules, s_nodules = splitTextures(nodules, metadata)
    
    s_train, s_test, s_val = splitData(s_nodules, [2 for i in range(len(s_nodules))])
    ss_train, ss_test, ss_val = splitData(ss_nodules, [1 for i in range(len(ss_nodules))])
    ns_train, ns_test, ns_val = splitData(ns_nodules,[0 for i in range(len(ns_nodules))])
    
    x_test = np.concatenate((ns_test[0], ss_test[0], s_test[0]), axis = 0)
    y_test = np.concatenate((ns_test[1], ss_test[1], s_test[1]), axis = 0)
    masks_test = assignMasks(x_test, masks)
    
    if mode == "default":
        x_train = np.concatenate((ns_train[0], ss_train[0], s_train[0]), axis = 0)
        y_train = np.concatenate((ns_train[1], ss_train[1], s_train[1]), axis = 0)
        
        x_val = np.concatenate((ns_val[0], ss_val[0], s_val[0]), axis = 0)
        y_val = np.concatenate((ns_val[1], ss_val[1], s_val[1]), axis = 0)
    
        masks_train = assignMasks(x_train, masks)
        masks_val = assignMasks(x_val, masks) 

    elif mode == "cross_val":
        x_ns = np.concatenate((ns_train[0],ns_val[0]), axis = 0)
        y_ns = np.concatenate((ns_train[1],ns_val[1]), axis = 0)
        
        x_ss = np.concatenate((ss_train[0],ss_val[0]), axis = 0)
        y_ss = np.concatenate((ss_train[1],ss_val[1]), axis = 0)
        
        x_s = np.concatenate((s_train[0],s_val[0]), axis = 0)
        y_s = np.concatenate((s_train[1],s_val[1]), axis = 0)
        
        n_splits = 5
       
        all_x_train_ns = []
        all_x_val_ns = []
        all_y_train_ns = []
        all_y_val_ns = []
        
        ns_kf = KFold(n_splits, shuffle = True)
        for train_index, val_index in ns_kf.split(x_ns):
            x_train_ns, x_val_ns = x_ns[train_index], x_ns[val_index]
            y_train_ns, y_val_ns = y_ns[train_index], y_ns[val_index]
            all_x_train_ns.append(x_train_ns)
            all_x_val_ns.append(x_val_ns)
            all_y_train_ns.append(y_train_ns)
            all_y_val_ns.append(y_val_ns)
            
        all_x_train_ss = []
        all_x_val_ss = []
        all_y_train_ss = []
        all_y_val_ss = []

        ss_kf = KFold(n_splits, shuffle = True)
        for train_index, val_index in ss_kf.split(x_ss):
            x_train_ss, x_val_ss = x_ss[train_index], x_ss[val_index]
            y_train_ss, y_val_ss = y_ss[train_index], y_ss[val_index]
            all_x_train_ss.append(x_train_ss)
            all_x_val_ss.append(x_val_ss)
            all_y_train_ss.append(y_train_ss)
            all_y_val_ss.append(y_val_ss)
        
        all_x_train_s = []
        all_x_val_s = []
        all_y_train_s = []
        all_y_val_s = []
        
        s_kf = KFold(n_splits, shuffle = True)
        for train_index, val_index in s_kf.split(x_s):
            x_train_s, x_val_s = x_s[train_index], x_s[val_index]
            y_train_s, y_val_s = y_s[train_index], y_s[val_index]
            all_x_train_s.append(x_train_s)
            all_x_val_s.append(x_val_s)
            all_y_train_s.append(y_train_s)
            all_y_val_s.append(y_val_s)
        
        x_train = [np.concatenate((all_x_train_ns[i], all_x_train_ss[i], all_x_train_s[i]), axis = 0) for i in range(n_splits)]
        y_train = [np.concatenate((all_y_train_ns[i], all_y_train_ss[i], all_y_train_s[i]), axis = 0) for i in range(n_splits)]
        
        x_val = [np.concatenate((all_x_val_ns[i], all_x_val_ss[i], all_x_val_s[i]), axis = 0) for i in range(n_splits)]
        y_val = [np.concatenate((all_y_val_ns[i], all_y_val_ss[i], all_y_val_s[i]), axis = 0) for i in range(n_splits)]
       

        masks_train = []
        masks_val = []
        for i in range(len(x_train)):
            masks_train.append(assignMasks(x_train[i], masks))
            masks_val.append(assignMasks(x_val[i], masks))
        
    return x_train, y_train, masks_train, x_test, y_test, masks_test, x_val, y_val, masks_val

"""
Get Middle Slice
================
Returns the middle slice of an volume (in this case a cube)

Arguments:
    * volume
    
Returns:
    * image: middle slice
"""

def getMiddleSlice(volume):
    sh = volume.shape
    
    return volume[...,np.int(sh[-1]/2)]

def loadImages(images, masks):
    new_images = []
    new_masks = []

    for i in range(len(images)):
        new_images.append(np.load(images[i,1]))
        new_masks.append(np.load(masks[i,1]))
    
    return new_images, new_masks


def getImages(x_train, masks_train, x_test, masks_test, x_val, masks_val, mode = "default"):
            
    test_nods, test_masks = loadImages(x_test, masks_test)
    test_slices = []
    test_slices_masks = []
    for n in range(len(test_nods)):
        test_slices.append(getMiddleSlice(test_nods[n]))
        test_slices_masks.append(getMiddleSlice(test_masks[n]))
    
    if mode == "default":
        train_nods, train_masks = loadImages(x_train, masks_train)
        train_slices = []
        train_slices_masks = []
        for n in range(len(train_nods)):
            train_slices.append(getMiddleSlice(train_nods[n]))
            train_slices_masks.append(getMiddleSlice(train_masks[n]))
    
        val_nods, val_masks = loadImages(x_val, masks_val)
        val_slices = []
        val_slices_masks = []
        for n in range(len(val_nods)):
            val_slices.append(getMiddleSlice(val_nods[n]))
            val_slices_masks.append(getMiddleSlice(val_masks[n]))
        
    elif mode == "cross_val":
        train_slices = []
        train_slices_masks = []
        val_slices = []
        val_slices_masks = []
        for i in range(len(x_train)):
            train_nods, train_masks = loadImages(x_train[i], masks_train[i])
            t_slices = []
            t_slices_masks = []
            for n in range(len(train_nods)):
                t_slices.append(getMiddleSlice(train_nods[n]))
                t_slices_masks.append(getMiddleSlice(train_masks[n]))
            train_slices.append(t_slices)
            train_slices_masks.append(t_slices_masks)
            
        for i in range(len(x_val)):
            val_nods, val_masks = loadImages(x_val[i], masks_val[i])
            t_slices = []
            t_slices_masks = []
            for n in range(len(val_nods)):
                t_slices.append(getMiddleSlice(val_nods[n]))
                t_slices_masks.append(getMiddleSlice(val_masks[n]))
            val_slices.append(t_slices)
            val_slices_masks.append(t_slices_masks)
              
          
    return train_slices, train_slices_masks, test_slices, test_slices_masks, val_slices, val_slices_masks