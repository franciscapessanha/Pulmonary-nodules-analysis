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
from sklearn.model_selection import cross_val_score

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

      
def getFeaturePoints(features_n, features_nn, number_of_pixel):
    
    i_nodules, i_non_nodules = getIndexes(features_n,features_nn)
    
    i_n = [i_nodules[j] for j in range(number_of_pixel)]
    i_nn = [i_non_nodules[j] for j in range(number_of_pixel)]
    
    nodule_points = [features_n[:,i] for i in i_n]
    non_nodule_points = [features_nn[:,i] for i in i_nn]
    
    points = np.concatenate((nodule_points, non_nodule_points), axis = 0)
    labels = np.concatenate((np.ones(np.shape(nodule_points)[0]),np.zeros(np.shape(non_nodule_points)[0])), axis = 0)
    return points, labels

def mask_gt(train_slices, train_slices_masks):
    #aplicar a mascara do pulmao as imagens 
    lung_px=[]
    gt_px=[]
    t_p=[]
    f_n=[]
    
    for i in range(len(train_slices)):
        t_positive = 0
        f_negative = 0
        lung_mask = get_lung_mask(train_slices[i])
        gt_px.append(train_slices_masks[i][lung_mask==0])
        
        gt_new_px=[]
        
        if i == 0:
            gt_new_px = gt_px[i]
        else:
            gt_new_px = np.concatenate((gt_new_px, gt_px[i]), axis = 0)
        
        for j in range(len(gt_new_px)):
                        
            if gt_new_px[j]==1:
                f_negative = f_negative+1
            else:
                t_positive = t_positive+1
                
        t_p.append(t_positive)  
        f_n.append(f_negative)
        
        tp = sum(t_p)
        fn = sum(f_n)
        
    return(tp,fn)
    

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

def input_set(val_slices, val_slices_masks):

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
            mask_px = val_masks_new_slices[i]
        else:
            val_slices_px= np.concatenate((val_slices_px, val_new_slices[i]), axis = 0)
            si_px = np.concatenate((si_px, shape_index_new_mask[i]), axis = 0)
            cv_px = np.concatenate((cv_px , CV_mask_new[i]), axis = 0)
            Vmed_px = np.concatenate((Vmed_px, Vmed_mask_new[i]), axis = 0)
            mask_px = np.concatenate((mask_px,val_masks_new_slices[i]), axis = 0)
        

    val_set = np.asarray((val_slices_px, si_px, cv_px, Vmed_px))
    val_set = np.transpose(val_set)
    
    return(val_set, mask_px)
   
def confusion_matrix(predictions, labels, val_slices, val_slices_masks):
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
                
    tp, fn = mask_gt(val_slices, val_slices_masks)
    
    true_positives = true_positives + tp
    false_negatives = false_negatives + fn
    
    dice = (2*true_positives/(false_positives+false_negatives+(2*true_positives)))
    
    jaccard = (true_positives)/(true_positives+false_positives+false_negatives)
    
    matrix = np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    return dice, jaccard, matrix


#%%

points, labels = mainSegmentation(train_slices, train_slices_masks, 10)

model = SVC(decision_function_shape='ovo', class_weight='balanced')
model.fit(points,labels)
    
val_set, mask_px = input_set(test_slices, test_slices_masks)

predictions = model.predict(val_set)

dice, jaccard, conf_matrix = confusion_matrix(predictions, mask_px, test_slices, test_slices_masks)
print(conf_matrix, dice, jaccard)



data = train_slices+val_slices
labels = train_slices_masks + val_slices_masks

cross_set, cross_mask = input_set(data, labels)

#new_shape_data = np.reshape(data, (5814,51))
#new_shape_labels = np.reshape(labels, (5814,51))

scores = cross_val_score(model, cross_set, cross_mask, cv=2)
