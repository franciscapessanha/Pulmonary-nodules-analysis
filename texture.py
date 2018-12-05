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
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.decomposition import PCA
from skimage.filters import gabor, gabor_kernel

# Intensity features
#===================
def getIntensityFeatures(nodules, masks):
    intensity_features = []
    
    for nodule, mask in zip(nodules, masks):
   
        mean = np.mean(nodule[mask == 1])
        max_ = np.max(nodule[mask == 1])
        min_ = np.min(nodule[mask == 1])
        median = np.median(nodule[mask == 1])
        std = np.std(nodule[mask == 1])
        #entrop = np.mean(entropy(nodule,disk(3)))
        intensity_features.append([mean, std, median])
        
    return intensity_features

# LBP
#=====

def maskedLBP(nodules,masks,radius = 1,n_points = 8):
    all_lbp = []
    for nodule, mask in zip(nodules, masks):
        lbp = local_binary_pattern(nodule, n_points, radius, 'default') #'ror' for rotation invariant    
        all_lbp.append(lbp[mask == 1])
    
    return all_lbp

def my_hist(all_lbp):
    all_hist = []
    for lbp in all_lbp:
        n_bins = 256
        hist,_ = np.histogram(lbp,normed = True, bins=n_bins, range=(0, n_bins))
        all_hist.append(hist)
    return all_hist

def getTextureFeatures(nodules, masks):
    all_textures = maskedLBP(nodules, masks)
    all_hists = my_hist(all_textures)
    return all_hists

"""
# Gabor
#======

def gabor_filtering(nodules, masks):
    all_gabor = []
    for nodule, mask in zip(nodules,masks):
        gabor_nodule = []
        
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (6, 7):
                for frequency in (0.06, 0.07):
                    filt_real, filt_imag = gabor(nodule, frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                    gabor_nodule.append(filt_real[mask ==1])
        all_gabor.append(gabor_nodule)
    return all_gabor
"""


# SVM
#=======

train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()

train_int_features = getIntensityFeatures(train_slices, train_slices_masks)
train_text_features = np.vstack(getTextureFeatures(train_slices, train_slices_masks))
#train_all_gabor = gabor_filtering(train_slices, train_slices_masks)


val_int_features = getIntensityFeatures(val_slices, val_slices_masks)
val_text_features = getTextureFeatures(val_slices, val_slices_masks)

train_all_features = np.hstack((train_int_features, train_text_features))
val_all_features = np.hstack((val_int_features, val_text_features))

print("Intensity Features only \n=======================")
modelSVM = SVC(gamma='scale', decision_function_shape='ovo', class_weight='balanced')

pca = PCA(n_components=2)
pca_train = pca.fit_transform(train_int_features)
pca_val = pca.fit_transform(val_int_features)

modelSVM.fit(pca_train, y_train)
predictSVM = modelSVM.predict(pca_val)
accuracy = accuracy_score(y_val, predictSVM)
print("Accuracy SVM (val) = %.3f" % accuracy)

print("Texture Features only \n=======================")
modelSVM = SVC(gamma='scale', decision_function_shape='ovo', class_weight='balanced')

pca = PCA(n_components=3)
pca_train = pca.fit_transform(train_text_features)
pca_val = pca.fit_transform(val_text_features)

modelSVM.fit(pca_train, y_train)
predictSVM = modelSVM.predict(pca_val)
accuracy = accuracy_score(y_val, predictSVM)
print("Accuracy SVM (val) = %.3f" % accuracy)


print("Texture and Intensity Features\n=======================")
modelSVM = SVC(gamma='scale', decision_function_shape='ovo', class_weight='balanced')

pca = PCA(n_components=3)
pca_train = pca.fit_transform(train_all_features)
pca_val = pca.fit_transform(val_all_features)

modelSVM.fit(pca_train, y_train)
predictSVM = modelSVM.predict(pca_val)
accuracy = accuracy_score(y_val, predictSVM)
print("Accuracy SVM (val) = %.3f" % accuracy)


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