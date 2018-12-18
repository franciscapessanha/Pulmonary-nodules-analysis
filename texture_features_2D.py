import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

"""
Texture Features
==============================================================================

Extract Gabor Filter and LBP

Arguments:
    * train_x: training set
    * train_masks: ground truth for training set
    * val_x: validation set
    * val_masks: ground truth for validation set
    
Return:
    * train_gabor: gabor filter results for training set
    * val_gabor: gabor filter results for validation set
    * train_lbp: LBP results for training set
    * val_lbp: LBP results for validation set

"""

def getTextureFeatures(train_x, train_masks, val_x, val_masks,):
    train_gabor, val_gabor= getGaborFilter(train_x, train_masks, val_x, val_masks)
    train_lbp, val_lbp = getLBPFeatures(train_x, train_masks, val_x, val_masks, 1*3,8*3)
    
    return train_gabor, val_gabor, train_lbp, val_lbp

"""
LBP Calculation 
==============================================================================

Extract Gabor Filter and LBP

Arguments:
    * nodules: input nodules
    * masks: input masks
    * n_points: number of points
    * radius
    
Return:
    * all_lbp: all lbp pixeis
    * metrics_lbp: lbp metrics

"""

def calcLBP(nodules, masks, n_points, radius):
    all_lbp = []
    metrics_lbp = []
    for nodule, mask in zip(nodules, masks):
        kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1,1))
        erode_mask = cv.erode(mask,kernel_ellipse)
        lbp = local_binary_pattern(nodule, n_points, radius, 'uniform')
        all_lbp.append(lbp[erode_mask == 1])
        metrics_lbp.append([np.mean(lbp[erode_mask == 1]), np.std(lbp[erode_mask == 1])])
    return all_lbp, metrics_lbp

"""
LBP Histogram
==============================================================================

Arguments:
    * all_lbp
    * max_
    * min_
    
Return:
    * all_hist

"""

def calcHist(all_lbp, max_, min_):
    all_hist = []
    for lbp in all_lbp:
        n_bins = max_ + 1
        hist,_ = np.histogram(lbp,normed = True, bins=n_bins,range=(min_, max_))
        all_hist.append(hist)
    return all_hist

"""
Get LBP Features
==============================================================================

Uses the LBP calculating function to extract the values for LBP and the metrics

Arguments:
    * train_x
    * train_masks
    * val_x
    * val_masks
    * radius
    * n_points
    
Return:
    * train_metrics: metrics for the training set
    * val_metrics: metrics for the validation or test set

"""

def getLBPFeatures(train_x, train_masks, val_x, val_masks,radius = 1,n_points = 8):
    train_lbp, train_metrics = calcLBP(train_x, train_masks, n_points, radius)
    val_lbp, val_metrics = calcLBP(val_x, val_masks, n_points, radius)
  
    max_ = int(np.max(np.hstack(train_lbp)))
    min_ = int(np.min(np.hstack(train_lbp)))
    
    return train_metrics, val_metrics

"""
Gabor Filter (frequency and orientation) Features
===============================================================================

Use Gabor Filters as features to train the model in order to find frequency and orientation.

Arguments:
    * nodules
    * masks
    
Return:
    * filtered_ims: all gabor filters in a array

"""

def calculateGaborFilters(nodules, masks):
    filtered_ims = []
    #for i in range(len(slices)):
    for nodule, mask in zip(nodules, masks):
        for theta in (0,45,90,135):
            theta = theta / 180. * np.pi
            for sigma in (0.5, 1.5, 3.):
                for frequency in (0.3, 0.4, 0.5):
                    filt_real, filt_imag = gabor(nodule, frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                    filtered_ims.append(filt_real[mask == 1])
                    
    return filtered_ims

"""
Reshape Gabor
===============================================================================

Arguments:
    * filtered_ims
    * masks
    
Return:
    * gabor_results: results of the Gabor Filters

"""

def reshapeGabor(filtered_ims, nodules):
    gabor_results=[]
    for j in range(0,len(nodules)):
        each_img_nodule = filtered_ims[36*j:36*j+36]
        nodule_metrics = []
        for i in range(len(each_img_nodule)):
            nodule_metrics.append([np.mean(each_img_nodule[i]), np.std(each_img_nodule[i])])
        gabor_results.append(np.hstack(nodule_metrics))
         
    return gabor_results

"""
Get Gabor Filter
===============================================================================

Arguments:
    * train_x
    * train_masks
    * val_x
    * val_masks
    
Return:
    * train_gabor_features: gabor features to train the model 
    * val_gabor_features: gabor features for the validation set

"""

def getGaborFilter(train_x, train_masks, val_x, val_masks):
    filtered_ims_train = calculateGaborFilters(train_x, train_masks)
    filtered_ims_val = calculateGaborFilters(val_x, val_masks)
    
    train_gabor_features = reshapeGabor(filtered_ims_train, train_x)
    val_gabor_features = reshapeGabor(filtered_ims_val, val_x)

    
    return train_gabor_features, val_gabor_features