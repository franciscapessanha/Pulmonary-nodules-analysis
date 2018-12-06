# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:01:46 2018

@author: Margarida
"""

from get_data import getData
import numpy as np
from lung_mask import get_lung_mask
from hessian_based import getEigNodules, gaussianSmooth, getSI, getCV, getVmed
from sklearn.svm import SVC

def separateFeatures(sample_features, sample, mask):
   features_n = []
   features_nn = [] 
   
   for feature in sample_features:
       features_n.append(feature[mask == 1])
       lung_mask= get_lung_mask(sample) - mask
       features_nn.append(feature[lung_mask==1])
    
   return np.asarray(features_n), np.asarray(features_nn)

def getIndexes(features_n, features_nn):
    #ir buscar os indices de cada pixel
    i_nodules = np.asarray([j for j in range(len(features_n[0]))])
    i_non_nodules = np.asarray([j for j in range(len(features_nn[0]))])
    np.random.shuffle(i_nodules)
    np.random.shuffle(i_non_nodules)
    return i_nodules, i_non_nodules

      
def getFeaturePoints(features_n, features_nn, number_of_pixel):
    i_nodules, i_non_nodules = getIndexes(features_n,features_nn)

    i_n = [i_nodules[j] for j in range(number_of_pixel)]
    i_nn = [i_non_nodules[j] for j in range(number_of_pixel)]
    
    nodule_points = [features_n[:,i] for i in i_n]
    non_nodule_points = [features_nn[:,i] for i in i_nn]
    
    points = np.concatenate((nodule_points, non_nodule_points), axis = 0)
    labels = np.concatenate((np.ones(np.shape(nodule_points)[0]),np.zeros(np.shape(non_nodule_points)[0])), axis = 0)
    return points, labels

def getTrainingSet(train_slices, train_slices_masks, number_of_pixel):
    smooth_img = gaussianSmooth(train_slices)
    eigen_nodules = getEigNodules(smooth_img)
    shape_index = getSI (eigen_nodules)
    CV_nodules = getCV(eigen_nodules)
    Vmed_nodules = getVmed(eigen_nodules)
    
    for i in range(len(train_slices)):
        sample_features = [train_slices[i], shape_index[i], CV_nodules[i], Vmed_nodules[i]]
        sample = train_slices[i]
        mask = train_slices_masks[i]
    
        features_n, features_nn = separateFeatures(sample_features, sample, mask)
    
        if i == 0:
            points, labels = getFeaturePoints(features_n, features_nn, number_of_pixel)
        else:
            p, l = getFeaturePoints(features_n, features_nn, number_of_pixel)
            points = np.concatenate((points, p), axis = 0)
            labels = np.concatenate((labels, l), axis = 0)
            
    return points, labels

def getInputSet(nodules, masks):
    smooth_img = gaussianSmooth(nodules)
    eigen_nodules = getEigNodules(smooth_img)
    SI_nodules = getSI(eigen_nodules)
    CV_nodules = getCV(eigen_nodules)
    Vmed_nodules = getVmed(eigen_nodules)
    
    masked_nodules = []
    masked_SI = []
    masked_CV = []
    masked_Vmed = []
    masked_gt = []
    
    for nodule, si, cv, v_med, mask in zip(nodules, SI_nodules, CV_nodules, Vmed_nodules, masks) :
        lung_mask = get_lung_mask(nodule)
        masked_nodules.append(nodule[lung_mask == 1])
        masked_SI.append(si[lung_mask == 1])
        masked_CV.append(cv[lung_mask == 1])
        masked_Vmed.append(v_med[lung_mask == 1])
        masked_gt.append(mask[lung_mask == 1])
        
    for i in(range(len(masked_nodules))):
        if i == 0:
            nodules_px = masked_nodules[i]
            si_px = masked_SI[i]
            cv_px = masked_CV[i]
            Vmed_px = masked_Vmed[i]
            mask_px = masked_gt[i]
        else:
            nodules_px= np.concatenate((nodules_px, masked_nodules[i]), axis = 0)
            si_px = np.concatenate((si_px, masked_SI[i]), axis = 0)
            cv_px = np.concatenate((cv_px , masked_CV[i]), axis = 0)
            Vmed_px = np.concatenate((Vmed_px, masked_Vmed[i]), axis = 0)
            mask_px = np.concatenate((mask_px,masked_gt[i]), axis = 0)
    
    input_set = np.transpose(np.asarray((nodules_px, si_px, cv_px, Vmed_px)))
    
    return (input_set , mask_px)
   
def outerLungPrediction(nodules, masks):
    labels_outer_lung = []
    predictions_outer_lung = []
    for nodule, mask in zip(nodules, masks):
        lung_mask = get_lung_mask(nodule)
        
        labels_outer_lung.append(mask[lung_mask==0])
        predictions_outer_lung.append(np.zeros(np.shape(nodule[lung_mask==0])))
    
    return np.hstack(predictions_outer_lung), np.hstack(labels_outer_lung)


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


def getPerformanceMetrics(predictions_lung, labels_lung, predictions_outer_lung, labels_outer_lung):
    c_matrix_lung = confusionMatrix(predictions_lung, labels_lung)
    c_matrix_outer_lung = confusionMatrix(predictions_outer_lung, labels_outer_lung)
    
    true_positives = c_matrix_lung[0,0] + c_matrix_outer_lung[0,0]
    false_negatives = c_matrix_lung[0,1] + c_matrix_outer_lung[0,1]
    false_positives = c_matrix_lung[1,0] + c_matrix_outer_lung[1,0]
    true_negatives = c_matrix_lung[1,1] + c_matrix_outer_lung[1,1]
 
    dice = (2*true_positives/(false_positives+false_negatives+(2*true_positives)))
    jaccard = (true_positives)/(true_positives+false_positives+false_negatives)
    matrix = np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    return dice, jaccard, matrix


train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()

points, labels = getTrainingSet(train_slices, train_slices_masks, 10)
model = SVC(gamma = 'scale', decision_function_shape='ovo', class_weight='balanced')
model.fit(points,labels)

val_lung, labels_lung  = getInputSet(val_slices, val_slices_masks)
predictions_lung = model.predict(val_lung)

predictions_outer_lung, labels_outer_lung = outerLungPrediction(val_slices, val_slices_masks)
dice, jaccard, matrix = getPerformanceMetrics(predictions_lung, labels_lung, predictions_outer_lung, labels_outer_lung)

