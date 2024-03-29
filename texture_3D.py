#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:13:57 2018

@author: mariafranciscapessanha
"""

from get_data import getData
from sklearn.svm import SVC
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from int_features_3D import getIntensityFeatures
from texture_features_3D import getTextureFeatures
from sklearn import metrics
"""
Run
==============

Allows to run the code for 2D texture, choosing default or cross-validation mode. 

Arguments:
    * mode: default or cross-validation
    
Return: void

"""
def run(mode = "default"):
    if mode == "default": 
        train_volumes, train_masks,y_train, val_volumes, val_masks,y_val, test_volumes, test_masks, y_test = getData(mode = "default", type_ = "volume")
        int_metrics, circ_metrics, lbp_metrics, gb_metrics, all_metrics, int_kNN_metrics, circ_kNN_metrics, lbp_kNN_metrics, gb_kNN_metrics, all_kNN_metrics = getTexture(train_volumes, train_masks,y_train, val_volumes, val_masks,y_val, test_volumes, test_masks, y_test)
        
    elif mode == "cross_val":
        int_metrics = []
        circ_metrics = []
        lbp_metrics = []
        gb_metrics = []
        all_metrics = []
        
        test_int_metrics = []
        test_circ_metrics = []
        test_lbp_metrics = []
        test_gb_metrics = []
        test_all_metrics = []
        
        cv_train_x, cv_train_masks, cv_train_y , cv_val_x, cv_val_masks, cv_val_y, test_x, test_masks, test_y = getData(mode = "cross_val", type_ = "volume")
        for train_x, train_masks, train_y, val_x, val_masks, val_y in zip(cv_train_x, cv_train_masks,cv_train_y, cv_val_x, cv_val_masks, cv_val_y):
            
           int_m, circ_m, lbp_m, gb_m, all_m, test_int_m, test_circ_m, test_lbp_m, test_gb_m, test_all_m = getTexture(train_x, train_masks, train_y, val_x, val_masks, val_y, test_x, test_masks, test_y)
           
           int_metrics.append(int_m)
           circ_metrics.append(circ_m)
           lbp_metrics.append(lbp_m)
           gb_metrics.append(gb_m)
           all_metrics.append(all_m)
           
           test_int_metrics.append(test_int_m)
           test_circ_metrics.append(test_circ_m)
           test_lbp_metrics.append(test_lbp_m)
           test_gb_metrics.append(test_gb_m)
           test_all_metrics.append(test_all_m)
            
        print("---------------VALIDATION SET --------------")
        showResultsCrossVal(int_metrics, circ_metrics, lbp_metrics,gb_metrics, all_metrics)
        print("---------------TEST SET --------------")
        showResultsCrossVal(test_int_metrics, test_circ_metrics, test_lbp_metrics, test_gb_metrics, test_all_metrics)
       
  """
Cross-Validation Results
==============

Prints the results for the cross-validation metrics.

Arguments:
    * int_metrics: intensity metrics for a classifier
    * circ_metris: circularity metrics for a classifier
    * lpb_metrics: LBP metrics for a classifier
    * gb_metrics: Gabor Filters metrics for a classifier
    * all_metrics: concatenate metrics for a classifier
    
Return: void
    
"""      
         
        
def showResultsCrossVal(int_metrics, circ_metrics, lbp_metrics,gb_metrics, all_metrics):
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

Normalization of the input data. 

Arguments:
    * train_slices: images related to a set of images
    * train_slices_masks: masks related to a set of images
    
Return:
    * mean_int: mean of all pixels of the nodules 
    * std_int: standard deviation of all pixels of the nodules 

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
Get Prediction SVM
===============================================================================

Train the model using the SVM classifier and predict.  

Arguments:
    * train_features: features related to the train set
    * train_y: labels of the training set
    * features: features of test or validation set
    * labels: labels of the test or validation set
        
Return:
    * predictSVM: prediction of the model 
    
"""

def getPredictionSVM(train_features, train_y, val_features, val_y):
    modelSVM = SVC(kernel = 'linear', gamma = 'auto', decision_function_shape= 'ovo',class_weight='balanced')
    modelSVM.fit(train_features, train_y)
    predictSVM = modelSVM.predict(val_features)

    return predictSVM

"""
Get Prediction kNN
===============================================================================

Train the model using the kNN classifier and predict.  

Arguments:
    * train_features: features related to the train set
    * train_y: labels of the training set
    * features: features of test or validation set
    * labels: labels of the test or validation set
        
Return:
    * predictkNN: prediction of the model 
    
"""

def getPredictionKNN(train_features, train_y, features, labels):
    modelKNN = KNeighborsClassifier(n_neighbors=11)
    modelKNN.fit(train_features, train_y)
    predictKNN = modelKNN.predict(features)

    return predictKNN

"""
Confusion Matrix
===============================================================================

Calculates the confusion matrix given a prediction and a label

Arguments:
    * predictions: results obtained from the classifier
    * labels: ground truth of the classification
        
Return:
    * true_positives
    * false_negatives
    * false_positives
    * true_negatives
    
"""

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


"""
getPerformanceMetrics
=========================

Calculates accuracy, precision, recall, auc for evaluation given an array of predictions and the corresponding ground truth

Arguments: 
    * predictions- array with predicted results
    * labels-  corresponding ground true

Return: 
    * accuracy 
    * precision 
    * recall 
    * auc 
"""
    
def getPerformanceMetrics(predictions, labels):
    c_matrix = confusionMatrix(predictions, labels)
    
    true_positives = c_matrix[0,0]
    false_negatives = c_matrix[0,1]
    false_positives = c_matrix[1,0]
    true_negatives = c_matrix[1,1]

    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    precision = (true_positives)/(true_positives + false_positives + 10**-12)
    
    recall = (true_positives)/(true_positives + false_negatives)
    #matrix = np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    fp_rate, tp_rate, thresholds = metrics.roc_curve(labels, predictions, pos_label = 1)
    auc = metrics.auc(fp_rate, tp_rate)
    
    return accuracy, precision, recall, auc

"""
separateClasses 
============================
Separates classes in 3 vector, one for each classes, where the 1's correspondes to the predicted true results and 0's to false predicted results                    

Arguments: 
        * predictSVM: array with multiple classes 

Returns: 
        * solid, sub_solid, non_solid - binary vectors of each class
"""

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

"""
Texture Metrics
===============================================================================

Train the model using the SVM classifier and predict.  

Arguments:
    * prediction: results obtained from the classifier
    * val_y: ground truth of the classification
        
Return:
    * true_positives
    * false_negatives
    * false_positives
    * true_negatives
    
"""

def textureMetrics(prediction, val_y):
    solid_pred, sub_solid_pred, non_solid_pred = separateClasses(prediction)
    solid_label, sub_solid_label, non_solid_label = separateClasses(val_y)
    
    accuracy_solid, precision_solid, recall_solid, auc_solid = getPerformanceMetrics(solid_pred, solid_label)
    accuracy_sub_solid, precision_sub_solid, recall_sub_solid, auc_sub_solid = getPerformanceMetrics(sub_solid_pred, sub_solid_label)
    accuracy_non_solid, precision_non_solid, recall_non_solid, auc_non_solid = getPerformanceMetrics(non_solid_pred, non_solid_label)
    print("Solid texture: The accuracy value is %.2f. The precision value is %.2f. The recall is %.2f. The AUC is %.2f" % (accuracy_solid, precision_solid, recall_solid, auc_solid))
    print("Sub solid texture: The accuracy value is %.2f. The precision value is %.2f. The recall is %.2f. The AUC is %.2f" % (accuracy_sub_solid, precision_sub_solid, recall_sub_solid, auc_sub_solid))
    print("Non solid texture: The accuracy value is %.2f. The precision value is %.2f. The recall is %.2f. The AUC is %.2f" % ( accuracy_non_solid, precision_non_solid, recall_non_solid, auc_non_solid))
   
    return [accuracy_solid, precision_solid, recall_solid, auc_solid, accuracy_sub_solid, precision_sub_solid, recall_sub_solid, auc_sub_solid,accuracy_non_solid, precision_non_solid, recall_non_solid, auc_non_solid]

"""
Get Texture
===============================================================================

Train the model using the SVM classifier and predict.  

Arguments:
    * train_x: training set
    * train_masks: ground truth of the training set
    * train_y: labels of the training set
    * val_x: validation or test set
    * val_masks: ground truth of the validation or test set
    * val_y: labels of the validation or test set
        
Return:
    * int_metrics: intensity metrics for validation
    * circ_metrics: circularity metrics for validation
    * lbp_metrics: LBP metrics for validation
    * gb_metrics: Gabor Filter metrics for validation
    * all_metrics: concatenate metrics of all features for validation
    * test_int_metrics: intensity metrics for test
    * test_circ_metrics: circularity metrics for test
    * test_lbp_metrics: LBP metrics for test
    * test_gb_metrics: Gabor Filter metrics for test
    * test_all_metrics: concatenate metrics of all features for test

"""  

def getTexture(train_x, train_masks, train_y, val_x, val_masks, val_y, test_x, test_masks, test_y):
    mean_int, std_int = normalizeData(train_x, train_masks)
    train_x = (train_x - mean_int)/std_int
    val_x = (val_x - mean_int)/std_int
    test_x = (test_x - mean_int)/std_int
    
    print("1. Intensity Features")
    train_int, val_int, test_int, train_circ, val_circ, test_circ = getIntensityFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    print("2. Texture Features")

    print("1. Intensity features")
    train_int, val_int, test_int, train_circ, val_circ, test_circ = getIntensityFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    print("2. Texture features")

    train_gabor, val_gabor, test_gabor,train_lbp, val_lbp, test_lbp = getTextureFeatures(train_x, train_masks, val_x, val_masks, test_x, test_masks)
    
    print("------------------------------------------- VALIDATION SET -------------------------------------------")
    
    print("\nIntensity Features only \n=======================")
    prediction_int = getPredictionSVM(train_int ,train_y, val_int , val_y)
    int_metrics = textureMetrics(prediction_int, val_y)
    
    print("\nCircular Features only \n=======================")
    prediction_circ = getPredictionSVM(train_circ, train_y, val_circ, val_y)
    circ_metrics = textureMetrics(prediction_circ, val_y)
    
    print("\nLBP Features only \n=======================")
    prediction_lbp = getPredictionSVM(train_lbp, train_y, val_lbp, val_y)
    lbp_metrics = textureMetrics(prediction_lbp, val_y)
    
    print("\nGabor Features only \n=======================")
    prediction_gb = getPredictionSVM(train_gabor, train_y, val_gabor, val_y)
    gb_metrics = textureMetrics(prediction_gb, val_y)
    
    print("\nAll Features\n=======================")
    train_features = np.concatenate((train_int, train_circ,train_lbp, train_gabor), axis=1)
    val_features = np.concatenate((val_int, val_circ, val_lbp, val_gabor), axis=1)
    
    prediction_all = getPredictionSVM(train_features, train_y, val_features, val_y)
    all_metrics = textureMetrics(prediction_all, val_y)
       
    
    print("------------------------------------------- TEST SET -------------------------------------------")
    
    print("\nIntensity Features only \n=======================")
    prediction_int = getPredictionSVM(train_int ,train_y, test_int , test_y)
    test_int_metrics = textureMetrics(prediction_int, test_y)
    
        
    print("\nCircular Features only \n=======================")
    prediction_circ = getPredictionSVM(train_circ, train_y, test_circ, test_y)
    test_circ_metrics = textureMetrics(prediction_circ, test_y)
    
    print("\nLBP Features only \n=======================")
    prediction_lbp = getPredictionSVM(train_lbp, train_y, test_lbp, test_y)
    test_lbp_metrics = textureMetrics(prediction_lbp, test_y)
    
        
    print("\nGabor Features only \n=======================")
    prediction_gb = getPredictionSVM(train_gabor, train_y, test_gabor, test_y)
    test_gb_metrics = textureMetrics(prediction_gb, test_y)
    
    print("\nAll Features\n=======================")
    train_features = np.concatenate((train_int, train_circ,train_gabor,train_lbp), axis=1)
    test_features = np.concatenate((test_int, test_circ, test_gabor,test_lbp), axis=1)
    prediction_all = getPredictionSVM(train_features, train_y, test_features, test_y)
    test_all_metrics = textureMetrics(prediction_all, test_y)
       
    
    return int_metrics, circ_metrics, lbp_metrics, gb_metrics, all_metrics, test_int_metrics, test_circ_metrics, test_lbp_metrics, test_gb_metrics, test_all_metrics
    
run()