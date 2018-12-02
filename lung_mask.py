# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:29:57 2018

@author: Hugo Barros
"""

import cv2
from skimage.morphology import convex_hull_image
import numpy as np
from matplotlib import pyplot as plt


"""
get_lung_mask
===============
creates a mask of the lungs where it exists. 

Arguments:
    * nodule_image: an image of a nodule
    
Returns:
    * chull: image of the mask after applied the convex hull - boolean type matrix
"""

def get_lung_mask(nodule_image):
    # global thresholding
    ret1,th1 = cv2.threshold(nodule_image,0.57,255,cv2.THRESH_BINARY) # binary treshold
    
    th1=(255-th1) #invert image

    #morphoogic operation - closing and opening
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(th1,kernel,iterations = 3)
    mask = cv2.erode(mask,kernel,iterations = 3)
    
    mask= cv2.erode(mask,kernel,iterations = 2)
    mask=cv2.dilate(mask,kernel,iterations = 2)
    
    chull = convex_hull_image(mask)
  
    return chull


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
def show_lung_mask(original, chull):
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
