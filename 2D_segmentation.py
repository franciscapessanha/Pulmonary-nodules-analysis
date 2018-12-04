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


smooth_img = gaussianSmooth(train_slices)
eigen_nodules = getEigNodules(smooth_img)
shape_index = getSI (eigen_nodules)
CV_nodules = getCV(eigen_nodules)
Vmed_nodules = getVmed(eigen_nodules)

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
def getFeaturePoints(features_n, features_nn):
    i_nodules, i_non_nodules = getIndexes(features_n,features_nn)
    
    i_n = [i_nodules[j] for j in range(10)]
    i_nn = [i_non_nodules[j] for j in range(10)]
    
    nodule_points = [features_n[:,i] for i in i_n]
    non_nodule_points = [features_nn[:,i] for i in i_nn]
    
    points = np.concatenate((nodule_points, non_nodule_points), axis = 0)
    labels = np.concatenate((np.ones(np.shape(nodule_points)[0]),np.zeros(np.shape(non_nodule_points)[0])), axis = 0)
    return points, labels
    
for i in range(len(train_slices)):
    sample_features = [train_slices[i], shape_index[i], CV_nodules[i], Vmed_nodules[i]]
    sample = train_slices[i]
    mask = train_slices_masks[i]
    
    features_n, features_nn = separateFeatures(sample_features, sample, mask)
    
    if i == 0:
        points, labels = getFeaturePoints(features_n, features_nn)
    else:
        p, l = getFeaturePoints(features_n, features_nn)
        points = np.concatenate((points, p), axis = 0)
        labels = np.concatenate((labels, l), axis = 0)
    

#%%
model = SVC(gamma='scale', decision_function_shape='ovo', class_weight='balanced')
model.fit(points,labels)
model.predict()