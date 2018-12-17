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
    

    return train_gabor, val_gabor, test_gabor,train_lbp, val_lbp, test_lbp
"""
LBP Features
==============================================================================
"""

def calcLBP(nodules, masks, n_points, radius):
    all_lbp = []
    metrics_lbp = []
    for nodule, mask in zip(nodules, masks):
        sample_lbp = []
        for slice_ in range(len(nodule)):
            if np.sum(mask[slice_,:,:]) != 0:
                kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1,1))
                erode_mask = cv.erode(mask[slice_,:,:],kernel_ellipse)
                lbp = local_binary_pattern(nodule[slice_,:,:], n_points, radius, 'uniform')
                sample_lbp.append(lbp[erode_mask == 1])
        metrics_lbp.append([np.mean(np.hstack(sample_lbp)), np.std(np.hstack(sample_lbp))])
        all_lbp.append(sample_lbp)
    return all_lbp, metrics_lbp

def getLBPFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks, radius = 1,n_points = 8):
    train_lbp, train_metrics = calcLBP(train_x, train_masks, n_points, radius)
    val_lbp, val_metrics = calcLBP(val_x, val_masks, n_points, radius)
    test_lbp, test_metrics = calcLBP(test_x, test_masks, n_points, radius)

    return train_metrics, val_metrics, test_metrics


"""
Gabor Filter (frequency and orientation) Features
===============================================================================
"""
def calculateGaborFilters(nodules, masks):
    filtered_ims = []
    slices_per_nodule = []
    for nodule, mask in zip(nodules, masks):
        n_slices = 0
        for slice_ in range(len(nodule)):
            if np.sum(mask[slice_,:,:]) != 0:
                n_slices +=1
                for theta in (0,45,90,135):
                    theta = theta / 180. * np.pi
                    for sigma in (0.5, 1.5, 3.):
                        for frequency in (0.3, 0.4, 0.5):
                            filt_real, filt_imag = gabor(nodule[slice_,:,:], frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                            filtered_ims.append(filt_real[mask[slice_,:,:] == 1])
        slices_per_nodule.append(n_slices)
    return filtered_ims,slices_per_nodule

def reshapeGabor(filtered_ims, nodules, slices_per_nodule):
    gabor_results=[]

    
    for j in range(len(nodules)):
        n_slices = slices_per_nodule[j]
        first = int(np.sum(slices_per_nodule[0:j]) * 36)
        last = int(np.sum(slices_per_nodule[0:j]) * 36 + 36 * n_slices)
        
        each_img_nodule = filtered_ims[first:last]
        nodule_metrics = []
       
        for i in range(36):
            values = [each_img_nodule[k] for k in range(i,len(each_img_nodule)-(35-i), 36)]
            nodule_metrics.append([np.mean(np.hstack(values)), np.std(np.hstack(values))])
            print(np.shape(nodule_metrics))
         
        gabor_results.append(nodule_metrics)
         
    return gabor_results

def getGaborFilter(train_x, train_masks, val_x, val_masks, test_x, test_masks):
    filtered_ims_train, slices_per_nodule_train = calculateGaborFilters(train_x, train_masks)
    filtered_ims_val, slices_per_nodule_val = calculateGaborFilters(val_x, val_masks)
    filtered_ims_test, slices_per_nodule_test = calculateGaborFilters(test_x, test_masks)
    
    train_gabor_features = reshapeGabor(filtered_ims_train, train_x,slices_per_nodule_train)
    val_gabor_features = reshapeGabor(filtered_ims_val, val_x,slices_per_nodule_val)
    test_gabor_features = reshapeGabor(filtered_ims_test, test_x,slices_per_nodule_test)
    
    return train_gabor_features, val_gabor_features, test_gabor_features
