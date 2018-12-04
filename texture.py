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

def getIntensityFeatures(nodules, masks):
    intensity_features = []
    
    for i in range(len(nodules)):
        nodule = nodules[i]
        #nodule = gaussian_filter(nodule, 1)
        mask = masks[i]
    
        mean = np.mean(nodule[mask != 0])
        max_ = np.max(nodule[mask != 0])
        min_ = np.min(nodule[mask != 0])
        median = np.median(nodule[mask != 0])
        std = np.std(nodule[mask != 0])
        intensity_features.append([mean, max_, min_, median, std])

    return intensity_features

def results(labels, predictions):
    TP_solid = 0
    TP_sub_solid = 0
    TP_non_solid = 0
    
    #solid instead of non_solid
    s_instead_ns = 0
    s_instead_ss = 0
   
    ss_instead_ns = 0
    ss_instead_s = 0
    
    ns_instead_ss = 0
    ns_instead_s = 0
    
    """
    labels:
        solid = 2
        sub_solid = 1
        non_solid = 0
    """
    
    for i in range(len(predictions)):
        
        if predictions[i] == labels[i] :
            if predictions[i] == 0:
                TP_non_solid += 1
            elif predictions[i] == 1:
                TP_sub_solid += 1
            elif predictions[i] == 2:
                TP_solid += 1   
        else:
            #prediction = non solid
            if predictions[i] == 0:
                if labels[i] == 1:
                    ns_instead_ss += 1
                elif labels[i] == 2:
                    ns_instead_s += 1            
            #prediction = sub solid
            if predictions[i] == 1:
                if labels[i] == 0:
                    ss_instead_ns += 1
                elif labels[i] == 2:
                    ss_instead_s += 1
            #prediction = solid
            if predictions[i] == 2:
                if labels[i] == 0:
                    s_instead_ns += 1
                elif labels[i] == 1:
                    s_instead_ss += 1
                
    true_positives = [TP_non_solid, TP_sub_solid, TP_solid]
    false_solids = [s_instead_ns, s_instead_ss]
    false_sub_solids = [ss_instead_ns, ss_instead_s]
    false_non_solids = [ns_instead_ss, ns_instead_s]
    
    return true_positives, false_solids, false_sub_solids, false_non_solids

train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()

# SVM
#=======

from sklearn.decomposition import PCA

modelSVM = SVC(gamma='scale', decision_function_shape='ovo', class_weight='balanced')
train_int_features = getIntensityFeatures(train_slices, train_slices_masks)
pca = PCA(n_components=3)
pca_train = pca.fit_transform(train_int_features)

#means = [train_int_features[i][0] for i in range(len(train_int_features))]
#stds = [train_int_features[i][1] for i in range(len(train_int_features))]
#plt.scatter(means, stds, c = y_train)
#plt.show()
modelSVM.fit(pca_train, y_train)

val_int_features = getIntensityFeatures(val_slices, val_slices_masks)
pca = PCA(n_components=3)
pca_val = pca.fit_transform(val_int_features)
predictSVM = modelSVM.predict(pca_val)
accuracy = accuracy_score(y_val, predictSVM)
print("Accuracy SVM (val) = %.3f" % accuracy)
val_int_features = getIntensityFeatures(val_slices, val_slices_masks)
pca = PCA(n_components=3)
pca_val = pca.fit_transform(val_int_features)


test_int_features = getIntensityFeatures(test_slices,test_slices_masks)
pca = PCA(n_components=3)
pca_test = pca.fit_transform(test_int_features)
predictSVM = modelSVM.predict(pca_test)
accuracy = accuracy_score(y_test, predictSVM)
print("Accuracy SVM (test) = %.3f" % accuracy)

# KNN
#======
from sklearn.neighbors import KNeighborsClassifier

for i in [3,5,7,11, 13, 17]:
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(train_int_features, y_train)
    predictkNN = neigh.predict(val_int_features)
    accuracy = accuracy_score(y_val, predictkNN)
    print("Accuracy for k = %.0f = %.3f" % (i, accuracy))

#true_positives, false_solids, false_sub_solids, false_non_solids = results(y_val, predictSVM)
