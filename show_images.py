#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:43:17 2018

@author: mariafranciscapessanha
"""
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries

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