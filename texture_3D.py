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
from sklearn.neighbors import KNeighborsClassifier
from int_features_3D import getIntensityFeatures
from texture_features_3D import getTextureFeatures
from sklearn import metrics
"""
Run
===============================================================================
""" 
def run(mode = "default"):
    if mode == "default": 
        train_volumes, train_masks,y_train, val_volumes, val_masks,y_val, test_volumes, test_masks, y_test = getData(mode = "default", type_ = "volume")
        int_metrics, circ_metrics, lbp_metrics, gb_metrics, all_metrics, int_kNN_metrics, circ_kNN_metrics, lbp_kNN_metrics, gb_kNN_metrics, all_kNN_metrics = getTexture(train_volumes[20:32], train_masks,y_train[20:32], val_volumes[1:10], val_masks,y_val[1:10], test_volumes[1:10], test_masks[1:10], y_test[1:10])
        
    elif mode == "cross_val":
        int_metrics = []
        circ_metrics = []
        lbp_metrics = []
        gb_metrics = []
        all_metrics = []
        
        int_kNN_metrics = []
        circ_kNN_metrics = []
        lbp_kNN_metrics = []
        gb_kNN_metrics = []
        all_kNN_metrics = []
        
        cv_train_x, cv_train_masks, cv_train_y , cv_val_x, cv_val_masks, cv_val_y, test_x, test_masks, test_y = getData(mode = "cross_val", type_ = "volume")
        for train_x, train_masks, train_y, val_x, val_masks, val_y in zip(cv_train_x, cv_train_masks,cv_train_y, cv_val_x, cv_val_masks, cv_val_y):
            int_metrics, circ_metrics, lbp_metrics, gb_metrics, all_metrics, int_kNN_metrics, circ_kNN_metrics, lbp_kNN_metrics, gb_kNN_metrics, all_kNN_metrics = getTexture(train_x, train_masks, train_y, val_x, val_masks, val_y, test_x, test_masks, test_y)
           
            
            print("SVM\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("\nIntensity Features only \n=======================")
            performaceCrossVal(int_metrics)
            print("\nCircular Features\n=======================")
            performaceCrossVal(circ_metrics)
            print("\nLBP Features only \n=======================")
            performaceCrossVal(lbp_metrics)
            print("\nGabor Features only \n=======================")
            performaceCrossVal(gb_metrics)
            print("\nAll Features\n=======================")
            performaceCrossVal(all_metrics)
            
            print("kNN\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("\nIntensity Features only \n=======================")
            performaceCrossVal(int_kNN_metrics)
            print("\nCircular Features\n=======================")
            performaceCrossVal(circ_kNN_metrics)
            print("\nLBP Features only \n=======================")
            performaceCrossVal(lbp_kNN_metrics)
            print("\nGabor Features only \n=======================")
            performaceCrossVal(gb_kNN_metrics)
            print("\nAll Features\n=======================")
            performaceCrossVal(all_kNN_metrics)

"""
Measure cross-validation performance
===============================================================================
"""   
def performaceCrossVal(metrics):
    metrics = np.vstack(metrics)
    mean = np.mean(metrics, axis = 0)
    std = np.std(metrics, axis = 0)
    
    print("Solid texture: The dice value is %.2f ± %.2f and the jaccard value is %.2f ± %.2f. The accuracy is %.2f ± %.2f" % 
          (mean[0], std[0],mean[1], std[1],mean[2],std[2]))
    print("Sub solid texture: The dice value is %.2f ± %.2f and the jaccard value is %.2f ± %.2f. The accuracy is %.2f ± %.2f" % 
          (mean[3], std[3],mean[4], std[4],mean[5],std[5]))
    print("Non solid texture: The dice value is %.2f ± %.2f and the jaccard value is %.2f ± %.2f. The accuracy is %.2f ± %.2f" % 
           (mean[6], std[6],mean[7], std[7],mean[8],std[8]))

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
def getPredictionSVM(train_features, train_y, val_features, val_y):
    modelSVM = SVC(kernel = 'linear', gamma = 'auto', decision_function_shape= 'ovo',class_weight='balanced')
    modelSVM.fit(train_features, train_y)
    predictSVM = modelSVM.predict(val_features)

    return predictSVM

def getPredictionKNN(train_features, train_y, features, labels):
    modelKNN = KNeighborsClassifier(n_neighbors=11)
    modelKNN.fit(train_features, train_y)
    predictKNN = modelKNN.predict(features)

    return predictKNN

def confusionMatrix(predictions, labels):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    
    for i in range(len(predictions)):
        if predictions[i] == labels[i] :
            if  predictions[i] == 1.0:
                true_positives += 1
            elif  predictions[i] == 0.0:
                true_negatives += 1
        elif predictions[i] != labels[i]:
            if predictions[i] == 1.0:
                false_positives += 1
            elif predictions[i] == 0.0:
                false_negatives += 1
                
    return np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]]) 
    
def getPerformanceMetrics(predictions, labels):
    c_matrix = confusionMatrix(predictions, labels)
    
    true_positives = c_matrix[0,0]
    false_negatives = c_matrix[0,1]
    false_positives = c_matrix[1,0]
    true_negatives = c_matrix[1,1]

    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    precision = (true_positives)/(true_positives + false_positives)
    recall = (true_positives)/(true_positives + false_negatives)
    matrix = np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    fp_rate, tp_rate, thresholds = metrics.roc_curve(labels, predictions, pos_label = 1)
    auc = metrics.auc(fp_rate, tp_rate)
    
    
    return accuracy, precision, recall, auc

def separateClasses(predictSVM):
    solid =[] # label 2
    sub_solid = [] # label 1
    non_solid = [] # label 0
    for j in range(len(predictSVM)):
        if predictSVM[j] == 0:
            non_solid.append(1)
        else: 
            non_solid.append(0)
            
        if predictSVM[j] == 1:
            sub_solid.append(1)
        else: 
            sub_solid.append(0)
            
        if predictSVM[j] == 2:
            solid.append(1)
        else: 
            solid.append(0)
            
    return solid, sub_solid, non_solid

def textureMetrics(prediction, val_y):
    solid_pred, sub_solid_pred, non_solid_pred = separateClasses(prediction)
    solid_label, sub_solid_label, non_solid_label = separateClasses(val_y)
    
    dice_solid, jaccard_solid, matrix_solid, accuracy_solid = getPerformanceMetrics(solid_pred, solid_label)
    dice_sub_solid, jaccard_sub_solid, matrix_sub_solid, accuracy_sub_solid = getPerformanceMetrics(sub_solid_pred, sub_solid_label)
    dice_non_solid, jaccard_non_solid, matrix_non_solid, accuracy_non_solid = getPerformanceMetrics(non_solid_pred, non_solid_label)
    
    print("Solid texture: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice_solid, jaccard_solid, accuracy_solid))
    print("Sub solid texture: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice_sub_solid, jaccard_sub_solid, accuracy_sub_solid))
    print("Non solid texture: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice_non_solid, jaccard_non_solid, accuracy_non_solid))


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
    train_gabor, val_gabor, test_gabor,train_lbp, val_lbp, test_lbp = getTextureFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    
    
    print("------------------------------------------- VALIDATION SET -------------------------------------------")
    
    print("\nIntensity Features only \n=======================")
    prediction_int = getPredictionSVM(train_int ,train_y, val_int , val_y)
    int_metrics = textureMetrics(prediction_int, val_y)
    prediction_kNN_int = getPredictionKNN(train_int ,train_y, test_int , test_y)
    int_kNN_metrics = textureMetrics(prediction_kNN_int, val_y)
    
    print("\nCircular Features only \n=======================")
    prediction_circ = getPredictionSVM(train_circ, train_y, val_circ, val_y)
    prediction_kNN_circ = getPredictionKNN(train_int ,train_y, test_int , test_y)
    circ_kNN_metrics = textureMetrics(prediction_kNN_circ, val_y)
    circ_metrics = textureMetrics(prediction_circ, val_y)
    
    print("\nLBP Features only \n=======================")
    prediction_lbp = getPredictionSVM(train_lbp, train_y, val_lbp, val_y)
    prediction_kNN_lbp = getPredictionKNN(train_int ,train_y, test_int , test_y)
    lbp_kNN_metrics = textureMetrics(prediction_kNN_lbp, val_y)
    lbp_metrics = textureMetrics(prediction_lbp, val_y)
    
    print("\nGabor Features only \n=======================")
    prediction_gb = getPredictionSVM(train_gabor, train_y, val_gabor, val_y)
    prediction_kNN_gb = getPredictionKNN(train_int ,train_y, test_int , test_y)
    gb_kNN_metrics = textureMetrics(prediction_kNN_gb, val_y)
    gb_metrics = textureMetrics(prediction_gb, val_y)
    
    print("\nAll Features\n=======================")
    train_features = np.concatenate((train_int, train_circ,train_lbp, train_gabor), axis=1)
    val_features = np.concatenate((val_int, val_circ, val_lbp, val_gabor), axis=1)
    test_features = np.concatenate((test_int, test_circ, test_lbp, test_gabor), axis=1)
    
    prediction_kNN_all = getPredictionKNN(train_int ,train_y, test_int , test_y)
    all_kNN_metrics = textureMetrics(prediction_kNN_all, val_y)
    
    prediction_all = getPredictionSVM(train_features, train_y, val_features, val_y)
    all_metrics = textureMetrics(prediction_all, val_y)
       
    """
    for i in range(len(test_x)):
        showImages([test_x[i]], [test_masks[i]], nodules_and_mask = True, overlay = False)
        
        print("Intensity = %.0f" % prediction_int[i])
        print("Circular = %.0f" % prediction_circ[i])
        print("LBP = %.0f" % prediction_lbp[i])
        print("Gabor = %.0f" % prediction_gb[i])
        print("All = %.0f" % prediction_all[i])
        print("GT = %.0f" % val_y[i])
    
    """ 
    
    """
    print("------------------------------------------- TEST SET -------------------------------------------")
    
    print("\nIntensity Features only \n=======================")
    prediction_int = getPredictionSVM(train_int ,train_y, test_int , test_y)
    prediction_KNN_int = getPredictionKNN(train_int ,train_y, test_int , test_y)
    print("\nSVM\n=======================")
    int_metrics = textureMetrics(prediction_int, test_y)
    print("\nkNN\n=======================")
    int_kNN_metrics = textureMetrics(prediction_KNN_int, test_y)
        
    print("\nCircular Features only \n=======================")
    prediction_circ = getPredictionSVM(train_circ, train_y, test_circ, test_y)
    prediction_KNN_circ = getPredictionKNN(train_circ ,train_y, test_circ, test_y)
    print("\nSVM\n=======================")
    circ_metrics = textureMetrics(prediction_circ, test_y)
    print("\nkNN\n=======================")
    circ_kNN_metrics = textureMetrics(prediction_KNN_circ, test_y)
    
    print("\nLBP Features only \n=======================")
    prediction_lbp = getPredictionSVM(train_lbp, train_y, test_lbp, test_y)
    prediction_KNN_lbp = getPredictionKNN(train_lbp ,train_y, test_lbp, test_y)
    print("\nSVM\n=======================")
    lbp_metrics = textureMetrics(prediction_lbp, test_y)
    print("\nkNN\n=======================")
    lbp_kNN_metrics = textureMetrics(prediction_KNN_lbp, test_y)
        
    print("\nGabor Features only \n=======================")
    prediction_gb = getPredictionSVM(train_gabor, train_y, test_gabor, test_y)
    prediction_KNN_gb = getPredictionKNN(train_gabor ,train_y, test_gabor, test_y)
    print("\nSVM\n=======================")
    gb_metrics = textureMetrics(prediction_gb, test_y)
    print("\nkNN\n=======================")
    gb_kNN_metrics = textureMetrics(prediction_KNN_gb, test_y)
      
    print("\nAll Features\n=======================")
    train_features = np.concatenate((train_int, train_circ,train_gabor,train_lbp), axis=1)
    test_features = np.concatenate((test_int, test_circ, test_gabor,test_lbp), axis=1)
    prediction_all = getPredictionSVM(train_features, train_y, test_features, test_y)
    prediction_kNN_all = getPredictionKNN(train_features, train_y, test_features, test_y)
    print("\nSVM\n=======================")
    all_metrics = textureMetrics(prediction_all, test_y)
    print("\nkNN\n=======================")
    all_kNN_metrics = predctictiontextureMetrics(prediction_kNN_all, test_y)
    """
    
    
    return int_metrics, circ_metrics, lbp_metrics, gb_metrics, all_metrics, int_kNN_metrics, circ_kNN_metrics, lbp_kNN_metrics, gb_kNN_metrics, all_kNN_metrics
    
run("cross_val")