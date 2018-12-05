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

train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()


#%%
samples = [[],[],[],[]]
labels = []


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

"""
def number_points (i_nodules, i_non_nodules):
    #definir o numero de pontos que vamos tirar de cada nodulo e de cada nao nodulo de uma imagem
    number_of_points_nodules = int(round(len(i_nodules)*0.2))
    number_of_points_non_nodules = int(round(len(i_non_nodules)*0.2))
    return number_of_points_nodules, number_of_points_non_nodules

"""      
def getFeaturePoints(features_n, features_nn, number_of_pixel):
    
    i_nodules, i_non_nodules = getIndexes(features_n,features_nn)
    
    i_n = [i_nodules[j] for j in range(number_of_pixel)]
    i_nn = [i_non_nodules[j] for j in range(number_of_pixel)]
    
    nodule_points = [features_n[:,i] for i in i_n]
    non_nodule_points = [features_nn[:,i] for i in i_nn]
    
    points = np.concatenate((nodule_points, non_nodule_points), axis = 0)
    labels = np.concatenate((np.ones(np.shape(nodule_points)[0]),np.zeros(np.shape(non_nodule_points)[0])), axis = 0)
    return points, labels

def confusion_matrix(predictions, labels):
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


def mainSegmentation(train_slices, train_slices_masks, number_of_pixel):
    
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
    

#%%

points,labels = mainSegmentation(train_slices, train_slices_masks, 10)

model = SVC(decision_function_shape='ovo', class_weight='balanced')
model.fit(points,labels)

smooth_img = gaussianSmooth(val_slices)
eigen_nodules = getEigNodules(smooth_img)
shape_index = getSI (eigen_nodules)
CV_nodules = getCV(eigen_nodules)
Vmed_nodules = getVmed(eigen_nodules)

val_new_slices = []
shape_index_new_mask = []
CV_mask_new = []
Vmed_mask_new = []
val_masks_new_slices = []

for i in range(len(shape_index)):
    lung_mask = get_lung_mask(val_slices[i])
    val_new_slices.append(val_slices[i][lung_mask==1])
    shape_index_new_mask.append(shape_index[i][lung_mask==1])
    CV_mask_new.append(CV_nodules[i][lung_mask==1])
    Vmed_mask_new.append(Vmed_nodules[i][lung_mask==1])
    val_masks_new_slices.append(val_slices_masks[i][lung_mask==1])
    
for i in(range(len(shape_index_new_mask))):
    if i == 0:
        val_slices_px = val_new_slices[i]
        si_px = shape_index_new_mask[i]
        cv_px = CV_mask_new[i]
        Vmed_px = Vmed_mask_new[i]
        Mask_px = val_masks_new_slices[i]
    else:
        val_slices_px= np.concatenate((val_slices_px, val_new_slices[i]), axis = 0)
        si_px = np.concatenate((si_px, shape_index_new_mask[i]), axis = 0)
        cv_px = np.concatenate((cv_px , CV_mask_new[i]), axis = 0)
        Vmed_px = np.concatenate((Vmed_px, Vmed_mask_new[i]), axis = 0)
        Mask_px = np.concatenate((Mask_px,val_masks_new_slices[i]), axis = 0)
        

val_set = np.asarray((val_slices_px, si_px, cv_px, Vmed_px))
val_set = np.transpose(val_set)

predictions = model.predict(val_set)

conf_matrix = confusion_matrix(predictions, Mask_px)
#print(conf_matrix)