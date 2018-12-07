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
from sklearn import linear_model 
from sklearn.neighbors import KNeighborsClassifier
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
Gabor Filter (frequency and orientation) Features
===================
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

def GetGaborFilter(train_slices, val_slices, test_slices):

    filtered_ims_train = calculateGaborFilters(train_slices)
    filtered_ims_val = calculateGaborFilters(val_slices)
    filtered_ims_test = calculateGaborFilters(test_slices)
    
    train_gabor_features = reshapeGabor(filtered_ims_train, train_slices)
    val_gabor_features = reshapeGabor(filtered_ims_val, val_slices)
    test_gabor_features = reshapeGabor(filtered_ims_test, test_slices)
    
    mean_gabor = np.mean(train_gabor_features, axis=0)
    std_gabor = np.std(train_gabor_features, axis=0)
    
    train_gabor_features = np.transpose([(train_gabor_features[:,i] - mean_gabor[i])/std_gabor[i] for i in range(len(mean_gabor))])
    val_gabor_features = np.transpose([(val_gabor_features[:,i] - mean_gabor[i])/std_gabor[i] for i in range(len(mean_gabor))])
    test_gabor_features = np.transpose([(test_gabor_features[:,i] - mean_gabor[i])/std_gabor[i] for i in range(len(mean_gabor))])
    
    return train_gabor_features, val_gabor_features, test_gabor_features

  
"""
SVM, PCA and plot
===================
"""
def getPrediction(train_features, y_train, val_features, y_val):
    modelSVM = SVC(gamma = 'auto', decision_function_shape='ovo', class_weight='balanced')
    modelSVM.fit(train_features, y_train)
    predictSVM = modelSVM.predict(val_features)
    
#kNN
    accuracys=[]
    prediction_knn = []
    """
    for k in [3,5,7,9,11,13]:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_features, y_train) 
        prediction_knn = neigh.predict(val_features)
        accuracy_knn = accuracy_score(y_val, prediction_knn)
        accuracys.append(accuracy_knn)
    """   
        
    accuracy = accuracy_score(y_val, predictSVM)
    print("Accuracy SVM = %.3f" % accuracy)

    """
    plt.scatter(pca_train[:,0], pca_train[:,1], c= y_train)
    plt.title("Training set")
    plt.show()
    plt.scatter(pca_val[:,0], pca_val[:,1], c= y_val)
    plt.title("Validation set")
    plt.show()
    """
    return predictSVM, prediction_knn, accuracys


#train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()
all_train_slices, all_train_slices_masks, all_y_train, test_slices, test_slices_masks, y_test , all_val_slices, all_val_slices_masks, all_y_val = getData("cross_val")
for train_slices, train_slices_masks, y_train, val_slices, val_slices_masks, y_val in zip(all_train_slices, all_train_slices_masks, all_y_train, all_val_slices, all_val_slices_masks, all_y_val): 
    
    train_int_features, val_int_features, test_int_features = getIntensityFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)
    train_shape_features, val_shape_features, test_shape_features = getShapeFeatures(train_slices_masks, val_slices_masks,test_slices_masks)
    train_lbp_features, val_lbp_features, test_lbp_features = getLBPFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)
    train_gabor_features, val_gabor_features, test_gabor_features = GetGaborFilter(train_slices, val_slices, test_slices)
    
    
    print("Intensity Features only \n=======================")
    prediction_int = getPrediction(train_int_features, y_train, val_int_features, y_val)
    
    print("Shape Features only \n=======================")
    prediction_shape = getPrediction(train_shape_features, y_train, val_shape_features, y_val)
    
    print("Texture Features only \n=======================")
    prediction_text = getPrediction(train_lbp_features, y_train, val_lbp_features, y_val)
    
    print("Gabor Features only \n=======================")
    prediction_gb = getPrediction(train_gabor_features, y_train, val_gabor_features, y_val)
    
    print("All Features\n=======================")
    train_features = np.concatenate((train_int_features, train_shape_features, train_lbp_features, train_gabor_features), axis=1)
    val_features = np.concatenate((val_int_features, val_shape_features, val_lbp_features, val_gabor_features), axis=1)
    test_features = np.concatenate((test_int_features, test_shape_features, test_lbp_features, test_gabor_features), axis=1)
    prediction_all = getPrediction(train_features, y_train, val_features, y_val)
    print("Test \n======")
    prediction_all = getPrediction(train_features, y_train, test_features, y_test)
    
    