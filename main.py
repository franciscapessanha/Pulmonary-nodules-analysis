#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:51:18 2018

@author: mariafranciscapessanha
"""
import numpy as np
from sklearn.model_selection import train_test_split
from get_started import loadData, showImages, meanIntensity, getMiddleSlice
nodules, masks, metadata = loadData()
nodules_indexes = [i for i in range(nodules.shape[1])]

#%%________________________________
# DIVIDING BY TEXTURE
#________________________________

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


def showAllImages():
    #show non_solid images
    print("NON SOLID \n==========")
    showImages(nodules, masks, non_solid, nodules_and_mask = False)
    
    #show sub_solid images
    print("SUB SOLID \n==========")
    showImages(nodules, masks, sub_solid, nodules_and_mask = False)
    
    #show sub_solid images
    print("SOLID \n==========")
    showImages(nodules, masks, solid, nodules_and_mask = False)
    

#%%________________________________
# SPLIT DATA
#________________________________

#to garanty that the proportion continues to be the same as the one on the 
#dataset 
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

"""

ISTO ESTÁ MAL
============
def splitData(data, labels):
    x_train, xi_test, y_train, yi_test = train_test_split(data, labels, test_size = 0.3, random_state=0)
    xf_test, x_val, yf_test, y_val = train_test_split(xi_test, yi_test, test_size = 0.5, random_state=0)
    
    train = [x_train, y_train]
    test = [xf_test, yf_test]
    val =[x_val, y_val]
    return train, test, val

#NON SOLID
#=========
ns_nodules = [nodules[:,i] for i in non_solid]
ns_train, ns_test, ns_val = splitData(ns_nodules, non_solid)

#SUB SOLID
#=========
ss_nodules = [nodules[:,i] for i in sub_solid]
ss_train, ss_test, ss_val = splitData(ss_nodules, sub_solid)

#SOLID
#=========
s_nodules = [nodules[:,i] for i in solid]
s_train, s_test, s_val = splitData(s_nodules, solid)

x_train = np.concatenate((ns_train[0], ss_train[0], s_train[0]), axis = 0)
y_train = np.concatenate((ns_train[1], ss_train[1], s_train[1]), axis = 0)

x_test = np.concatenate((ns_test[0], ss_test[0], s_test[0]), axis = 0)
y_test = np.concatenate((ns_test[1], ss_test[1], s_test[1]), axis = 0)

x_val = np.concatenate((ns_val[0], ss_val[0], s_val[0]), axis = 0).reshape(-1,1)
y_val = np.concatenate((ns_val[1], ss_val[1], s_val[1]), axis = 0)

def assignMasks(data, masks):
    new_masks = []
    for i in range(len(data)):
        for j in range(masks.shape[1]):
            if masks[0,j] == data[i,0]:
                new_masks.append(masks[:,j])

    return np.asarray(new_masks)

masks_train = assignMasks(x_train, masks)
masks_test = assignMasks(x_test, masks)
masks_val = assignMasks(x_val, masks)
"""

#%%________________________________
# SEGMENTATION - Campilho 
#________________________________
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpimg
import numpy as np

from scipy.ndimage import gaussian_filter

# 1. Multiscale Gaussian smoothing using sigm in the range 0.5 to 3.5
# ====================================================================
sigma = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
smooth_nodules = []
splited_nodules = []
for i in range(nodules.shape[1]):
    nodule = getMiddleSlice(np.load(nodules[1,i]))
    splited_nodules.append(nodule)
    smooth_nod = []
    for s in range(len(sigma)):
        smooth_nod.append(gaussian_filter(nodule, sigma[s]))
    
    smooth_nodules.append(smooth_nod)
    
# testing if it worked
ex= 3
plot_args={}
plot_args['vmin']=0
plot_args['vmax']=1
plot_args['cmap']='gray'

sample = smooth_nodules[ex]
print("ORIGINAL \n==========")
original_plot = plt.imshow(getMiddleSlice(nodule[ex],**plot_args))
plt.show()
for j in range(len(sample)):
    print("SIGMA = %.1f \n===========" % sigma[j])
    smooth_plot = plt.imshow(sample[j],**plot_args)
    plt.show()

# 2. Compute the Hessian matrix
# =============================

from skimage.feature import hessian_matrix
#image = sample[1] - np.mean(sample[1])

image = sample[2]
Hrr, Hrc, Hcc = hessian_matrix(image)
lower_ev = image
higher_ev = image
eig_image = np.zeros((len(image),len(image))).tolist()

for i in range(len(image)):
    for j in range(len(image)):
        eig_values, _ = np.linalg.eig(np.asarray([[Hrr[i,j], Hrc[i,j]],[Hrc[i,j], Hcc[i,j]]]))
        sorted_ev = np.asarray(np.sort(eig_values))
        lower_ev[i,j] = sorted_ev[0]
        higher_ev[i,j] = sorted_ev[1]

def normalizePlanes(npzarray,maxHU=400.,minHU=-1000.):#400, -1200
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray


plot_args={}
plot_args['vmin']= np.min(lower_ev)
plot_args['vmax']= np.max(higher_ev)
plot_args['cmap']='gray'
lower_plot = plt.imshow(lower_ev, **plot_args)
plt.show()
higher_plot = plt.imshow(higher_ev, **plot_args)
plt.show()

plot_args={}
plot_args['vmin']=0
plot_args['vmax']=1
plot_args['cmap']='gray'
mask = np.load(masks_train[ex,1])
mask = getMiddleSlice(mask)
plt.imshow(mask, **plot_args)

