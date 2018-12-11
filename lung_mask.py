# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:29:57 2018

@author: Hugo Barros
"""

import cv2 as cv
from skimage.morphology import convex_hull_image
import numpy as np
from matplotlib import pyplot as plt
from get_data import getData
from show_images import showImages
from skimage.measure import label, regionprops


"""
get_lung_mask
===============
creates a mask of the lungs where it exists. 

Arguments:
    * nodule_image: an image of a nodule
    
Returns:
    * chull: image of the mask after applied the convex hull - boolean type matrix
"""

def getLungMask(nodule, type_ = "slice"):
    
    if type_ == "slice":
        nodule_mask = cv.inRange(nodule, 0, 0.57)
        kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        mask = cv.medianBlur(nodule_mask,3)
        dilated_mask = cv.dilate(mask,kernel_ellipse,iterations = 1)
        erode_mask = cv.erode(dilated_mask,kernel_ellipse,iterations = 2)    
        
        _, contours,_= cv.findContours(erode_mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv.contourArea, reverse = True) 
        if len(contours) > 0: 
            contour = contours[0]
            filled_contour = np.zeros(np.asarray(nodule).shape, np.uint8)
            cv.fillPoly(filled_contour, pts = np.asarray(contour), color = 1)
            hull = convex_hull_image(filled_contour)  
    
    elif type_ == "volume":
        hull = np.zeros((51,51,51), np.uint8)
        for slice_ in range(len(nodule)):

            nodule_mask = cv.inRange(nodule[slice_,:,:], 0, 0.57)
            kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
            mask = cv.medianBlur(nodule_mask,3)
            dilated_mask = cv.dilate(mask,kernel_ellipse,iterations = 1)
            erode_mask = cv.erode(dilated_mask,kernel_ellipse,iterations = 2)    
            
            _, contours,_= cv.findContours(erode_mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv.contourArea, reverse = True) 
            if len(contours) > 0: 
                contour = contours[0]
                filled_contour = np.zeros(np.asarray(nodule[slice_,:,:]).shape, np.uint8)
                cv.fillPoly(filled_contour, pts = np.asarray(contour), color = 1)        
                hull[slice_,:,:] = convex_hull_image(filled_contour) 
            

    return hull


"""
show_lung_mask
===============
Plots 2 images, for comparison 

Arguments:
    * original: image (intended to be teh original nodule image)
    * chull: image (intended to be the mask of lungs)
    
Returns:
    void
"""
def showLungMask(original, chull):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].set_title('Original picture')
    ax[0].imshow(original, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_axis_off()
    
    #chull = cv2.erode(chull, kernel, iterations = 1)
    ax[1].set_title('Convex Hull')
    ax[1].imshow(chull, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_axis_off()
    
    plt.tight_layout()
    plt.show()
