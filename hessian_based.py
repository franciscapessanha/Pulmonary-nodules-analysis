#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:56:01 2018

@author: mariafranciscapessanha
"""

from get_data import getData
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numpy import linalg
import cv2 as cv
from skimage.feature import shape_index
from show_images import showImages

# 1. Multiscale Gaussian smoothing using sigm in the range 0.5 to 3.5
# ====================================================================

def gaussianSmooth(nodules):
    sigma = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    smooth_nodules = []
    for nodule in nodules:
        smooth_node = []
        for s in range(len(sigma)):
             smooth_node.append(gaussian_filter(nodule, sigma[s]))
  
        smooth_nodules.append(smooth_node)
        #plotImage(gaussian_filter(nodule, sigma[s]))
    
    return smooth_nodules

# 2. Compute the Hessian matrix and eig values
# ============================================
"""
Hessian matrix:
    
Describes the 2nd order local image intensity variations around the selected 
voxel. For the obtained Hessian matrix, eigenvector decomposition extracts an 
orthonormal coordinate system that is aligned with the second order structure 
of the image. Having the eigenvalues and knowing the (assumed) model of the 
structure to be detected and the resulting theoretical behavior of the eigenvalues, 
the decision can be made if the analyzed voxel belongs to the structure being searched.

Eigen values:

Eigenvalues give information about a matrix; the Hessian matrix contains geometric 
information about thesurface z = f(x, y). We’re going to use the eigenvalues of 
the Hessian matrix to get geometric information about the surface   
"""

"""
The eigenvalues of the Hessian matrix, in decreasing order. The eigenvalues are 
the leading dimension. That is, eigs[i, j, k] contains the ith-largest eigenvalue 
at position (j, k).
"""
def getEig(image):
    Hxx, Hxy, Hyy = hessian_matrix(image, order="xy")
    return hessian_matrix_eigvals(Hxx, Hxy, Hyy)

def plotEig(nodule_eig):
    lower_eig = nodule_eig[0][1]
    higher_eig = nodule_eig[0][0]
    plot_args={}
    plot_args['vmin']= np.min(lower_eig)
    plot_args['vmax']= np.max(higher_eig)
    plot_args['cmap']='gray'
    plt.imshow(lower_eig, **plot_args)
    plt.show()
    plt.imshow(higher_eig, **plot_args)
    plt.show()
  

def getEigNodules(smooth_nodules):
    eig_nodules = []   
    for nodule in range(len(smooth_nodules)):
        all_nodule_sigmas = smooth_nodules[nodule]
        nodule_eig = []
        
        for s in range(len(all_nodule_sigmas)):
            eig_vals = getEig(all_nodule_sigmas[s])
            nodule_eig.append(eig_vals)
            #plotEig(eig_vals)
        eig_nodules.append(nodule_eig)
    
    return eig_nodules

# 3.0.1 Get best image
# ====================

def getMaximumResponse(eig_nodules):
    max_nodules = []
    
    for n in range(len(eig_nodules)):
        max_nodule_0 = np.zeros((51,51))
        max_nodule_1 = np.zeros((51,51))
        all_nodule_sigmas = eig_nodules[n] 
        
        for i in range(51):
            for j in range(51):
                values_0 = []
                values_1 = []
                for s in range(len(all_nodule_sigmas)):
                    
                    nodule_0 = all_nodule_sigmas[s][0]
                    nodule_1 = all_nodule_sigmas[s][1]
                    px_0= nodule_0[i,j]
                    px_1= nodule_1[i,j]
                    
                    values_0.append(px_0)
                    values_1.append(px_1)
                
                max_nodule_0[i][j] = np.max(values_0) 
                max_nodule_1[i][j] = np.min(values_1)   
                
        #plotImage(max_nodule_0)        
        #plotImage(max_nodule_1) 
        max_nodules.append([max_nodule_0, max_nodule_1])
        
        return max_nodules

# 3.1 Shape index 
# ===============
    
"""
The shape index, as defined by Koenderink & van Doorn [1], is a single valued
measure of local curvature, assuming the image as a 3D plane with intensities 
representing heights.
It is derived from the eigen values of the Hessian, and its value ranges from 
-1 to 1 (and is undefined (=NaN) in flat regions), with following ranges 
representing following shapes
"""
def plotImage(image):
    plot_args={}
    plot_args['vmin']= np.min(image)
    plot_args['vmax']= np.max(image)
    plot_args['cmap']='gray'
    plt.imshow(image, **plot_args)
    plt.show()

def getSI(max_nodules):
    SI_nodules = [] 
    for nodule in max_nodules:
        lower_eig = nodule[1]
        higher_eig = nodule[0]
        shape_indexes = (2/np.pi) * np.arctan((lower_eig + higher_eig)/(lower_eig - higher_eig))
        #plotImage(shape_indexes)
        SI_nodules.append(shape_indexes)
        
    return SI_nodules


# 3.2 Curvedness approach 
# ======================== 
#we will compute the curvedness manually

def getCV(max_nodules):
    CV_nodules = []    
    for nodule in max_nodules:
        lower_eig = nodule[1]
        higher_eig = nodule[0]
        curvedness = np.sqrt(lower_eig**2 + higher_eig**2)
        #plotImage(curvedness)
        CV_nodules.append(curvedness)
    
    return CV_nodules
    
# 3.3 Central adaptive miedialness approach 
# ==========================================

def getVmed(max_nodules):
    Vmed_nodules = []
    for nodule in max_nodules:
        lower_eig = nodule[1]
        higher_eig = nodule[0] 
        Vmed = np.zeros((51,51))
        for i in range(len(Vmed)):
            for j in range(len(Vmed)):
                if lower_eig[i][j] + higher_eig[i][j] < 0:
                    Vmed[i][j] = -(lower_eig[i][j]/lower_eig[i][j]) * (lower_eig[i][j] + lower_eig[i][j])
        #plotImage(Vmed)
        Vmed_nodules.append(Vmed)
        
    return Vmed_nodules

# 3.4 Run and plot all
# =====================

train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()
smooth_nodules = gaussianSmooth(train_slices)
eig_nodules = getEigNodules(smooth_nodules)
max_nodules = getMaximumResponse(eig_nodules)
SI_nodules = getSI(max_nodules)
CV_nodules = getCV(max_nodules) 
Vmed_nodules = getVmed(max_nodules) 

for i in range(len(train_slices)):
    showImages([train_slices[i]], [train_slices_masks[i]],overlay = False)
    plotImage(SI_nodules[i])
    plotImage(CV_nodules[i])
    plotImage(Vmed_nodules[i])


#%%
#4 Combination of the results - thresholding (tentar mascara pulmonar)
# ====================================================================
# SI - thresholding
# -----------------
t_SI = 0.4
SI_nodules_t = []
for nodule in SI_nodules:
    thresh_SI = np.zeros((51,51))
    for i in range(len(thresh_SI)):
        for j in range(len(thresh_SI)):   
            if nodule[i][j] >= t_SI:
                thresh_SI[i][j] = 1

    #plotImage(thresh_SI) 
    SI_nodules_t.append(thresh_SI)
  
# CV - thresholding
# -----------------
t_CV = 0.05
CV_nodules_t = []
for nodule in CV_nodules:

    thresh_CV = np.zeros((51,51))
    for i in range(len(thresh_CV)):
        for j in range(len(thresh_CV)):   
            if nodule[i][j] >= t_CV:
                thresh_CV[i][j] = 1
    
    #plotImage(thresh_CV) 
    CV_nodules_t.append(thresh_CV)

# Vmed - thresholding
# -----------------
t_Vmed = 0.05
Vmed_nodules_t = []

Vmed_nodules_t = []
for nodule in Vmed_nodules:

    thresh_Vmed = np.zeros((51,51))
    for i in range(len(thresh_Vmed)):
        for j in range(len(thresh_Vmed)):   
            if nodule[i][j] >= t_Vmed:
                thresh_Vmed[i][j] = 1
    
    #plotImage(thresh_Vmed) 
    CV_nodules_t.append(thresh_Vmed)
    

    