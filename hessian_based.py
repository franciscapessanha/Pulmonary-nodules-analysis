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
from skimage.feature import shape_index
from show_images import showImages
from sklearn.neighbors import KNeighborsClassifier


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

#%%%

def getSI(eig_nodules):
    SI_nodules = [] 
    
    for all_sigmas_nodule in eig_nodules:
        all_SI = []
        for s in range(len(all_sigmas_nodule)):
            nodule = all_sigmas_nodule[s]
            lower_eig = nodule[1]
            higher_eig = nodule[0]
            shape_indexes = (2/np.pi) * np.arctan((lower_eig + higher_eig)/(lower_eig - higher_eig))    
            all_SI.append(shape_indexes)
        
        SI_nodule = np.zeros((51,51))
        for i in range(51):
            for j in range(51):
                values = []
                for k in range(len(all_sigmas_nodule)):
                    si = all_SI[k]
                    px = si[i,j]
                    values.append(px)
                SI_nodule[i][j] = np.max(values)
    
        SI_nodules.append(SI_nodule)
    
    return SI_nodules

#%%%
# 3.2 Curvedness approach 
# ======================== 
#we will compute the curvedness manually

def getCV(max_nodules):
    CV_nodules = []
    for all_sigmas_nodule in eig_nodules:
        all_CV = []
        for s in range(len(all_sigmas_nodule)):
            nodule = all_sigmas_nodule[s]
            lower_eig = nodule[1]
            higher_eig = nodule[0]
            curvedness = np.sqrt(lower_eig**2 + higher_eig**2)    
            all_CV.append(curvedness)
        
        CV_nodule = np.zeros((51,51))
        for i in range(51):
            for j in range(51):
                values = []
                for k in range(len(all_sigmas_nodule)):
                    cv_ = all_CV[k]
                    px = cv_[i,j]
                    values.append(px)
                CV_nodule[i][j] = np.max(values)
        CV_nodules.append(CV_nodule)
    return CV_nodules
    
# 3.3 Central adaptive miedialness approach 
# ==========================================

def getVmed(max_nodules):     
    Vmed_nodules = []
    for all_sigmas_nodule in eig_nodules:
        all_Vmed = []
        for s in range(len(all_sigmas_nodule)):
            nodule = all_sigmas_nodule[s]
            lower_eig = nodule[1]
            higher_eig = nodule[0]
            Vmed = np.zeros((51,51))
            for i in range(len(Vmed)):
                for j in range(len(Vmed)):
                    if lower_eig[i][j] + higher_eig[i][j] < 0:
                        Vmed[i][j] = -(lower_eig[i][j]/lower_eig[i][j]) * (lower_eig[i][j] + lower_eig[i][j])
            all_Vmed.append(Vmed)
        Vmed_nodule = np.zeros((51,51))
        for i in range(51):
            for j in range(51):
                values = []
                for k in range(len(all_sigmas_nodule)):
                    vmed = all_Vmed[k]
                    px = vmed[i,j]
                    values.append(px)
                Vmed_nodule[i][j] = np.max(values)
        Vmed_nodules.append(Vmed_nodule)
    return Vmed_nodules

# 3.4 Run and plot all
# =====================

train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()
    
smooth_nodules = gaussianSmooth(train_slices)
eig_nodules = getEigNodules(smooth_nodules)
SI_nodules = getSI(eig_nodules)
CV_nodules = getCV(eig_nodules) 
Vmed_nodules = getVmed(eig_nodules) 

for i in range(len(train_slices)):
    showImages([train_slices[i]], [train_slices_masks[i]],overlay = False)
    plotImage(SI_nodules[i])
    plotImage(CV_nodules[i])
    plotImage(Vmed_nodules[i])
