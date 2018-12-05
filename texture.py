#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:13:57 2018

@author: mariafranciscapessanha
"""

from get_data import getData
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.decomposition import PCA
from skimage.filters import gabor, gabor_kernel
import cv2 as cv

"""
Intensity Features
===================
"""
def normalizeIntensity(train_slices, train_slices_masks):
    all_px = []
    for nodule, mask in zip(train_slices, train_slices_masks):
        all_px.append(nodule[mask == 1])  
    all_px = np.hstack(all_px)
    mean_int = np.mean(all_px)
    std_int = np.std(all_px)
    return mean_int, std_int

def calcIntensityFeatures(nodules, masks):
    intensity_features = []
    for nodule, mask in zip(nodules, masks):
        mean = np.mean(nodule[mask == 1])
        max_ = np.max(nodule[mask == 1])
        min_ = np.min(nodule[mask == 1])
        median = np.median(nodule[mask == 1])
        std = np.std(nodule[mask == 1])
        intensity_features.append([mean, std, median, max_, min_])  
    
    return intensity_features
    

def getIntensityFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks):
    mean_int, std_int = normalizeIntensity(train_slices, train_slices_masks)

    train_nodules = [(nodule - mean_int)/std_int for nodule in train_slices]
    val_nodules = [(nodule - mean_int)/std_int for nodule in val_slices]
    test_nodules = [(nodule - mean_int)/std_int for nodule in test_slices]
    
    train_int_features = np.vstack(calcIntensityFeatures(train_nodules, train_slices_masks))
    val_int_features = np.vstack(calcIntensityFeatures(val_nodules, val_slices_masks))
    test_int_features = np.vstack(calcIntensityFeatures(test_nodules, test_slices_masks))
    
    return train_int_features, val_int_features, test_int_features

"""
Shape Features
===================
"""
def calcShapeFeatures(masks):
    shape_features = []
    for mask in masks:  
        _, contour, _ = cv.findContours(mask.astype(np.uint8),cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contour = contour[0]
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        eq_diameter = (2*np.pi)*np.sqrt(area)
        compactness = (perimeter**2)/ (4 * np.pi * area)
        circularity = 4 * np.pi * area / ( perimeter**2 )
        shape_features.append([area, perimeter, eq_diameter, compactness, circularity])
    return shape_features

def getShapeFeatures(train_slices_masks,val_slices_masks, test_slices_masks):
    train_shape_features = np.vstack(calcShapeFeatures(train_slices_masks))
    val_shape_features = np.vstack(calcShapeFeatures(val_slices_masks))
    test_shape_features = np.vstack(calcShapeFeatures(test_slices_masks))
    
    mean_shape = np.mean(train_shape_features, axis=0)
    std_shape = np.std(train_shape_features, axis=0)

    train_shape_features = np.transpose([(train_shape_features[:,i] - mean_shape[i])/std_shape[i] for i in range(len(mean_shape))])
    val_shape_features = np.transpose([(val_shape_features[:,i] - mean_shape[i])/std_shape[i] for i in range(len(mean_shape))])
    test_shape_features = np.transpose([(test_shape_features[:,i] - mean_shape[i])/std_shape[i] for i in range(len(mean_shape))])
    
    return train_shape_features, val_shape_features, test_shape_features
  

"""
LBP Features
===================
"""

def calcLBP(nodules, masks, n_points, radius):
    all_lbp = []
    for nodule, mask in zip(nodules, masks):
        lbp = local_binary_pattern(nodule, n_points, radius, 'default') #'ror' for rotation invariant    
        all_lbp.append(lbp[mask == 1])
    return all_lbp

def calcHist(all_lbp, max_, min_):
    all_hist = []
    for lbp in all_lbp:
        n_bins = int(256)
        
        hist,_ = np.histogram(lbp,normed = True, bins=n_bins, range = (int(min_), int(max_)))
        all_hist.append(hist)
    return all_hist

def getLBPFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks, radius = 1,n_points = 8):
    train_lbp = calcLBP(train_slices, train_slices_masks, n_points, radius)
    val_lbp = calcLBP(val_slices, val_slices_masks, n_points, radius)
    test_lbp = calcLBP(test_slices, test_slices_masks, n_points, radius)

    mean_lbp = np.mean(np.hstack(train_lbp))
    std_lbp = np.std(np.hstack(train_lbp))
    
    train_lbp = [(value - mean_lbp)/std_lbp for value in train_lbp]
    val_lbp = [(value - mean_lbp)/std_lbp for value in val_lbp]
    test_lbp = [(value - mean_lbp)/std_lbp for value in test_lbp]
    
    max_ = np.max([np.max(np.hstack(train_lbp)), np.max(np.hstack(val_lbp)), np.max(np.hstack(test_lbp))])
    min_ = np.min([np.min(np.hstack(train_lbp)), np.min(np.hstack(val_lbp)), np.min(np.hstack(test_lbp))])
    
    train_hist = np.vstack(calcHist(train_lbp, max_, min_))
    val_hist = np.vstack(calcHist(val_lbp,max_, min_))
    test_hist = np.vstack(calcHist(test_lbp, max_, min_))
    
    return train_hist, val_hist, test_hist

"""
SVM, PCA and plot
===================
"""
def getPrediction(train_features, y_train, val_features, y_val):
    modelSVM = SVC(gamma='scale', decision_function_shape='ovo', class_weight='balanced')

    pca = PCA(n_components= 2)
    pca_train = pca.fit_transform(train_features)
    pca_val = pca.fit_transform(val_features)

    modelSVM.fit(train_features, y_train)
    predictSVM = modelSVM.predict(val_features)
    accuracy = accuracy_score(y_val, predictSVM)
    print("Accuracy SVM (val) = %.3f" % accuracy)

    """
    plt.scatter(pca_train[:,0], pca_train[:,1], c= y_train)
    plt.title("Training set")
    plt.show()
    plt.scatter(pca_val[:,0], pca_val[:,1], c= y_val)
    plt.title("Validation set")
    plt.show()
    """
    return predictSVM

"""
Não usei
========
def getGLCMFeatures(nodules,masks):
    glcm_features = []
    for nodule, mask in zip(train_slices, masks): 
        nodule_glcm = np.copy(nodule)
        nodule_glcm[mask == 0] = None
        glcm = greycomatrix((nodule_glcm*255).astype(np.uint8), [5], [0], 256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, "contrast")
        dissimilarity = greycoprops(glcm, "dissimilarity")
        homogeneity = greycoprops(glcm, "homogeneity")
        asm = greycoprops(glcm, "ASM")
        energy = greycoprops(glcm, "energy")
        correlation = greycoprops(glcm, "correlation")
        glcm_features.append([contrast[0,0], dissimilarity[0,0], homogeneity[0,0], asm[0,0], energy[0,0], correlation[0,0]])
    
    return glcm_features
"""

train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()
train_int_features, val_int_features, test_int_features = getIntensityFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)
train_shape_features, val_shape_features, test_shape_features = getShapeFeatures(train_slices_masks, val_slices_masks,test_slices_masks)
train_lbp_features, val_lbp_features, test_lbp_features = getLBPFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)

print("Intensity Features only \n=======================")
prediction_int = getPrediction(train_int_features, y_train, val_int_features, y_val)

print("Shape Features only \n=======================")
prediction_shape = getPrediction(train_shape_features, y_train, val_shape_features, y_val)

print("Texture Features only \n=======================")
prediction_text = getPrediction(train_lbp_features, y_train, val_lbp_features, y_val)

print("All Features\n=======================")
train_features = np.concatenate((train_int_features, train_shape_features, train_lbp_features), axis=1)
val_features = np.concatenate((val_int_features, val_shape_features, val_lbp_features), axis=1)
test_features = np.concatenate((test_int_features, test_shape_features, test_lbp_features), axis=1)
prediction_all = getPrediction(train_features, y_train, val_features, y_val)