#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:42:19 2018

@author: mariafranciscapessanha
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.segmentation import find_boundaries
from sklearn.model_selection import train_test_split


#%%
#________________________________
# LOAD DATA
#________________________________

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

    
nodules, masks, metadata = loadData()
nodules_indexes = [i for i in range(nodules.shape[1])]

#%%
#________________________________
# SPLIT DATA
#________________________________

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

    x_train, xi_test, y_train, yi_test = train_test_split(data, labels, train_size = 20, random_state=0)
    xf_test, x_val, yf_test, y_val = train_test_split(xi_test, yi_test, test_size = 0.5, random_state=0)
    
    train = [x_train, y_train]
    test = [xf_test, yf_test]
    val =[x_val, y_val]
    return train, test, val


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



"""
labels:
    solid = 2
    sub_solid = 1
    non_solid = 0
"""

s_train, s_test, s_val = splitData(s_nodules, [2 for i in range(len(s_nodules))])
ss_train, ss_test, ss_val = splitData(ss_nodules, [1 for i in range(len(ss_nodules))])
ns_train, ns_test, ns_val = splitData(ns_nodules,[1 for i in range(len(ns_nodules))])


x_train = np.concatenate((ns_train[0], ss_train[0], s_train[0]), axis = 0)
y_train = np.concatenate((ns_train[1], ss_train[1], s_train[1]), axis = 0)

x_test = np.concatenate((ns_test[0], ss_test[0], s_test[0]), axis = 0)
y_test = np.concatenate((ns_test[1], ss_test[1], s_test[1]), axis = 0)

x_val = np.concatenate((ns_val[0], ss_val[0], s_val[0]), axis = 0)
y_val = np.concatenate((ns_val[1], ss_val[1], s_val[1]), axis = 0)

def assignMasks(set_, masks):
    new_masks = []
    for i in range(len(set_)):
        
        for j in range(masks.shape[1]):
            if masks[0,j] == set_[i,0]:
                
                new_masks.append(masks[:,j])

    return np.asarray(new_masks)

masks_train = assignMasks(x_train, masks)
masks_test = assignMasks(x_test, masks)
masks_val = assignMasks(x_val, masks)


#%%
#________________________________
# SHOW IMAGES
#________________________________

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

"""
Create Overlay
==============

Verifies if the mask and image have the same size. If so it will create a green
contour (0,1,0) around the nodule

Arguments:
    * im: image
    * mask: mask
    * color: color of the contour (green by default)
    * contour: bool, it true it will draw the contour on the image (true by default)
    
Return:
    * im: image
"""

def createOverlay(im,mask,color=(0,1,0),contour=True):
    if len(im.shape)==2:
        im = np.expand_dims(im,axis=-1)
        im = np.repeat(im,3,axis=-1)
    elif len(im.shape)==3:
        if im.shape[-1] != 3:
            ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)

    else:
        ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)
   
    if contour:
        bw = find_boundaries(mask,mode='thick') #inner
    else:
        bw = mask
    for i in range(0,3):
        im_temp = im[:,:,i]
        im_temp = np.multiply(im_temp,np.logical_not(bw)*1)
        im_temp += bw*color[i]
        im[:,:,i] = im_temp
    return im


"""
Show Images
================
Arguments:
    * nodules: numpy array with the the nodules names and paths
    * masks: numpy array  the the masks names and paths
    * nodules_indexes: indexes of the nodules we want to show
    * nodules_and_mask: show nodule and mask side by side (true by default)
    * overlay: show nodule and mask overlay (true by default)
     
"""

def showImages(nodules, masks, nodules_and_mask = True, overlay = True):
    plot_args={}
    plot_args['vmin']=0
    plot_args['vmax']=1
    plot_args['cmap']='gray'
    
    if nodules_and_mask:
        for i in range(len(nodules)):
            nodule = nodules[i]
            mask = masks[i]            
            #since we have volume we must show only a slice
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(nodule,**plot_args) #plots the image
            ax[1].imshow(mask,**plot_args) #plots the mask
            plt.show()
        
    #if instead you want to overlay
    if overlay:
        for n in range(len(nodules)):
            nodule = nodules[n]
            mask = masks[n]
            over = createOverlay(nodule,mask)
            #since we have volume we must show only a slice
            fig,ax = plt.subplots(1,1)
            ax.imshow(over,**plot_args)
            plt.show()
            

def loadImages(images, masks):
    new_images = []
    new_masks = []

    for i in range(len(images)):
        new_images.append(np.load(images[i,1]))
        new_masks.append(np.load(masks[i,1]))
    
    return new_images, new_masks

train_nods, train_masks = loadImages(x_train, masks_train)

train_slices = []
train_slices_masks = []
for n in range(len(train_nods)):
    train_slices.append(getMiddleSlice(train_nods[n]))
    train_slices_masks.append(getMiddleSlice(train_masks[n]))

showImages(train_slices,train_slices_masks)