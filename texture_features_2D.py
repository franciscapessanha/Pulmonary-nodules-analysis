import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor, gabor_kernel
import cv2 as cv
"""
Texture Features
==============================================================================
"""

def getTextureFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks):
    train_gabor, val_gabor, test_gabor = getGaborFilter(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    train_lbp, val_lbp, test_lbp = getLBPFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks, 1*3,8*3)
    
    #return train_gabor, val_gabor, test_gabor,train_lbp, val_lbp, test_lbp
    return train_gabor, val_gabor, test_gabor,train_lbp, val_lbp, test_lbp
"""
LBP Features
==============================================================================
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

def calcHist(all_lbp, max_, min_):
    all_hist = []
    for lbp in all_lbp:
        n_bins = max_ + 1
        
        hist,_ = np.histogram(lbp,normed = True, bins=n_bins,range=(min_, max_))
        all_hist.append(hist)
    return all_hist


def getLBPFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks, radius = 1,n_points = 8):
    train_lbp, train_metrics = calcLBP(train_x, train_masks, n_points, radius)
    val_lbp, val_metrics = calcLBP(val_x, val_masks, n_points, radius)
    test_lbp, test_metrics = calcLBP(test_x, test_masks, n_points, radius)

    max_ = int(np.max(np.hstack(train_lbp)))
    min_ = int(np.min(np.hstack(train_lbp)))
    """
    train_hist = np.vstack(calcHist(train_lbp, max_, min_))
    val_hist = np.vstack(calcHist(val_lbp, max_, min_))
    test_hist = np.vstack(calcHist(test_lbp, max_, min_))
    """
    
    #return train_hist, val_hist, test_hist
    return train_metrics, val_metrics, test_metrics


"""
Gabor Filter (frequency and orientation) Features
===============================================================================
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

def reshapeGabor(filtered_ims, nodules):
    gabor_results=[]
    
    for j in range(0,len(nodules)):
        each_img_nodule = filtered_ims[36*j:36*j+36]
        nodule_metrics = []
        for i in range(len(each_img_nodule)):
            nodule_metrics.append([np.mean(each_img_nodule[i]), np.std(each_img_nodule[i])])
        gabor_results.append(np.hstack(nodule_metrics))
         
            
    return gabor_results

def getGaborFilter(train_x, train_masks, val_x, val_masks, test_x, test_masks):
    filtered_ims_train = calculateGaborFilters(train_x, train_masks)
    filtered_ims_val = calculateGaborFilters(val_x, val_masks)
    filtered_ims_test = calculateGaborFilters(test_x, test_masks)
    
    train_gabor_features = reshapeGabor(filtered_ims_train, train_x)
    val_gabor_features = reshapeGabor(filtered_ims_val, val_x)
    test_gabor_features = reshapeGabor(filtered_ims_test, test_x)
    
    return train_gabor_features, val_gabor_features, test_gabor_features
