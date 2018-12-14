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
from sklearn.decomposition import PCA
from int_features_3D import getIntensityFeatures
from texture_features_3D import getTextureFeatures
"""
Run
===============================================================================
""" 
def run(mode = "default"):
    if mode == "default": 
        train_volumes, train_masks,y_train, val_volumes, val_masks,y_val, test_volumes, test_masks, y_test = getData(mode = "default", type_ = "volume")
        getTexture(train_volumes[0:4], train_masks[0:4],y_train[0:4], val_volumes[0:4], val_masks[0:4],y_val[0:4], test_volumes[0:4], test_masks[0:4], y_test[0:4])
        
    elif mode == "cross_val":
        cv_train_x, cv_train_masks, train_y , cv_val_x, cv_val_masks, val_y, test_x, test_masks, test_y = getData("cross_val")
        for train_x, train_masks, val_x, val_masks in zip(cv_train_x, cv_train_masks, cv_val_x, cv_val_masks):
            getTexture(train_x, train_masks, train_y, val_x, val_masks, val_y, test_x, test_masks, test_y)

    
"""
Normalize data
===============================================================================
"""  
def normalizeData(train_slices, train_slices_masks):
    all_px = []
    for nodule, mask in zip(train_slices, train_slices_masks):
        all_px.append(nodule[mask == 1])  
    all_px = np.hstack(all_px)
    mean_int = np.mean(all_px)
    std_int = np.std(all_px)
    
    return mean_int, std_int
 
"""
Get Prediction
===============================================================================
"""
def getPrediction(train_features, train_y, val_features, val_y):
    modelSVM = SVC(kernel = 'linear', gamma = 'auto', decision_function_shape= 'ovo',class_weight='balanced')
    modelSVM.fit(train_features, train_y)
    predictSVM = modelSVM.predict(val_features)

    accuracys=[]

    accuracy = accuracy_score(val_y, predictSVM)
    print("Accuracy SVM = %.3f" % accuracy)

    return predictSVM


"""
Get Texture
===============================================================================
""" 
def getTexture(train_x, train_masks, train_y, val_x, val_masks, val_y, test_x, test_masks, test_y):
    mean_int, std_int = normalizeData(train_x, train_masks)
    train_x = (train_x - mean_int)/std_int
    val_x = (val_x - mean_int)/std_int
    test_x = (test_x - mean_int)/std_int
    
    train_int, val_int, test_int, train_circ, val_circ, test_circ = getIntensityFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    #train_int, val_int, test_int = getIntensityFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    
    #train_shape, val_shape, test_shape = getShapeFeatures(train_masks,val_masks, test_masks)
    train_gabor, val_gabor, test_gabor,train_lbp, val_lbp, test_lbp = getTextureFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\nIntensity Features only \n=======================")
    prediction_int = getPrediction(train_int ,train_y, val_int , val_y)
    
    print("\nCircular Features only \n=======================")
    prediction_circ = getPrediction(train_circ, train_y, val_circ, val_y)
    
    print("\nLBP Features only \n=======================")
    prediction_lbp = getPrediction(train_lbp, train_y, val_lbp, val_y)
    
    print("\nGabor Features only \n=======================")
    prediction_gb = getPrediction(train_gabor, train_y, val_gabor, val_y)
    
    print("\nAll Features\n=======================")
    train_features = np.concatenate((train_int, train_lbp, train_gabor), axis=1)
    val_features = np.concatenate((val_int, val_lbp, val_gabor), axis=1)
    test_features = np.concatenate((test_int, test_lbp, test_gabor), axis=1)
    prediction_all = getPrediction(train_features, train_y, val_features, val_y)
    
    print("(PCA) All Features\n=======================")
    
    pca = PCA(n_components = 8)
    train_int_pca = np.concatenate((train_int, train_circ), axis = 1)
    val_int_pca = np.concatenate((val_int), axis = 1)
    pca.fit_transform(train_int_pca)
    pca.fit_transform(val_int_pca)
    
    pca = PCA(n_components = 8)
    train_text_pca = np.concatenate((train_lbp, train_gabor), axis = 1)
    val_text_pca = np.concatenate((val_lbp, val_gabor), axis = 1)
    pca.fit_transform(train_text_pca)
    pca.fit_transform(val_text_pca)
       
    prediction_all_pca = getPrediction(np.concatenate((train_int, train_text_pca), axis = 1), train_y,np.concatenate((val_int, val_text_pca), axis = 1), val_y)
    
    
    """
    for i in range(len(val_x)):
        showImages([val_x[i]], [val_masks[i]], nodules_and_mask = True, overlay = False)
    
        print("Intensity = %.0f" % prediction_int[i])
        print("Circular = %.0f" % prediction_circ[i])
        #print("Shape = %.0f" % prediction_shape[i])
        print("LBP = %.0f" % prediction_lbp[i])
        print("All = %.0f" % prediction_all[i])
        print("All (PCA) = %.0f" % prediction_all_pca[i])
        print("GT = %.0f" % val_y[i])
    """  

run()