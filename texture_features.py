import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor, gabor_kernel

"""
Texture Features
==============================================================================
"""

def getTextureFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks):
    #train_gabor, val_gabor, test_gabor = getGaborFilter(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    train_lbp, val_lbp, test_lbp = getLBPFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks, 3,8*3)
    
    #return train_gabor, val_gabor, test_gabor,train_lbp, val_lbp, test_lbp
    return train_lbp, val_lbp, test_lbp
"""
LBP Features
==============================================================================
"""

def calcLBP(nodules, masks, n_points, radius):
    all_lbp = []
    metrics_lbp = []
    for nodule, mask in zip(nodules, masks):
        lbp = local_binary_pattern(nodule, n_points, radius, 'var') #'ror' for rotation invariant    
        all_lbp.append(lbp[mask == 1])
        metrics_lbp.append([np.mean(lbp[mask == 1]), np.std(lbp[mask == 1])])
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
def calculateGaborFilters(slices):
    filtered_ims = []
    for i in range(len(slices)):
        for theta in range(3):
            theta = theta / 3. * np.pi
            for sigma in (6, 7):
                for frequency in (0.06, 0.07):
                    filt_real, filt_imag = gabor(slices[i], frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                    filtered_ims.append(filt_real)
    return filtered_ims

def reshapeGabor(filtered_ims, slices):
    vector_h_img=[]
    img_gabor_filters=[]
    all_img=[]
    for i in range(len(filtered_ims)):
        h_img = np.hstack(filtered_ims[i])
        vector_h_img.append(h_img)
    px_img = np.asarray(vector_h_img)
    for j in range(0,len(slices)):
        img_gabor_filters.append(px_img[12*j:12*j+12][:])
        h_img_gabor_filters = np.hstack(img_gabor_filters[j])
        all_img.append(h_img_gabor_filters)     
    
    gabor_features = np.asarray(all_img) 
    return gabor_features

def getGaborFilter(train_x, train_masks, val_x, val_masks, test_x, test_masks):
    filtered_ims_train = calculateGaborFilters(train_x)
    filtered_ims_val = calculateGaborFilters(val_x)
    filtered_ims_test = calculateGaborFilters(test_x)
    
    train_gabor_features = reshapeGabor(filtered_ims_train, train_x)
    val_gabor_features = reshapeGabor(filtered_ims_val, val_x)
    test_gabor_features = reshapeGabor(filtered_ims_test, test_x)
    
    return train_gabor_features, val_gabor_features, test_gabor_features

