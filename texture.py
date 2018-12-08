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
from skimage.measure import label, regionprops

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

def calcCircularFeatures(nodules, masks):
    circular_features = []
    i = 0
    for nodule, mask in zip(nodules, masks):
        print(i)
        circle_0 = np.zeros(np.shape(nodule), np.uint8)
        circle_1 = np.zeros(np.shape(nodule), np.uint8)
        circle_2 = np.zeros(np.shape(nodule), np.uint8)
        region = label(mask)
        props = regionprops(region)
        minor_axis = props[0]['minor_axis_length']
        centroid = props[0]['centroid']

        cv.circle(circle_0, (int(centroid[0]), int(centroid[1])),int(minor_axis*0.25),1, -1)
        cv.circle(circle_1, (int(centroid[0]), int(centroid[1])),int(minor_axis*0.50),1, -1)
        cv.circle(circle_2, (int(centroid[0]), int(centroid[1])),int(minor_axis*0.8),1, -1)
        
        
        mean_0 = np.mean(nodule[circle_0 == 1])
        mean_1 = np.mean(nodule[circle_1 == 1])
        mean_2 = np.mean(nodule[circle_2 == 1])
        
        std_0 = np.std(nodule[circle_0 == 1])
        std_1 = np.std(nodule[circle_1 == 1])
        std_2 = np.std(nodule[circle_2 == 1])
        
        #ent_0 = entropy(nodule/3,disk(5))
        #ent_1 = entropy(nodule/3,disk(5))
        #ent_2 = entropy(nodule/3,disk(5))
        
        circular_features.append([mean_0,mean_1, mean_2])
        i += 1
    return circular_features

def calcIntensityFeatures(nodules, masks):
    intensity_features = []
    for nodule, mask in zip(nodules, masks):
        mean = np.mean(nodule[mask == 1])
        #max_ = np.max(nodule[mask == 1])
        #min_ = np.min(nodule[mask == 1])
        #median = np.median(nodule[mask == 1])
        std = np.std(nodule[mask == 1])
        ent = entropy(nodule/3,disk(5))
        intensity_features.append([mean,std,np.mean(ent[mask == 1])])  
    
    return intensity_features
    
def getIntensityFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks):
    train_int_features = np.vstack(calcIntensityFeatures(train_slices, train_slices_masks))
    val_int_features = np.vstack(calcIntensityFeatures(val_slices, val_slices_masks))
    test_int_features = np.vstack(calcIntensityFeatures(test_slices, test_slices_masks))
    return train_int_features, val_int_features, test_int_features

def getCircularFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks):
    train_circ_features = np.vstack(calcCircularFeatures(train_slices, train_slices_masks))
    val_circ_features = np.vstack(calcCircularFeatures(val_slices, val_slices_masks))
    test_circ_features = np.vstack(calcCircularFeatures(test_slices, test_slices_masks))
    
    return train_circ_features, val_circ_features, test_circ_features
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
        compactness = (perimeter**2)/ (4 * np.pi * area)
        circularity = 4 * np.pi * area / ( perimeter**2 )
        shape_features.append([area, compactness, circularity])
    return shape_features

def getShapeFeatures(train_slices_masks,val_slices_masks, test_slices_masks):
    train_shape_features = np.vstack(calcShapeFeatures(train_slices_masks))
    val_shape_features = np.vstack(calcShapeFeatures(val_slices_masks))
    test_shape_features = np.vstack(calcShapeFeatures(test_slices_masks))
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
        n_bins = max_ + 1
        
        hist,_ = np.histogram(lbp,normed = True, bins=n_bins,range=(min_, max_))
        all_hist.append(hist)
    return all_hist

def getLBPFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks, radius = 4,n_points = 8*4):
    train_lbp = calcLBP(train_slices, train_slices_masks, n_points, radius)
    val_lbp = calcLBP(val_slices, val_slices_masks, n_points, radius)
    test_lbp = calcLBP(test_slices, test_slices_masks, n_points, radius)

    max_ = int(np.max(np.hstack(train_lbp)))
    min_ = int(np.min(np.hstack(train_lbp)))

    train_hist = np.vstack(calcHist(train_lbp, max_, min_))
    val_hist = np.vstack(calcHist(val_lbp, max_, min_))
    test_hist = np.vstack(calcHist(test_lbp, max_, min_))
    
    return train_hist, val_hist, test_hist


"""
Gabor Filter (frequency and orientation) Features
===================
"""

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
 
"""
SVM, PCA and plot
===================
"""
def getPrediction(train_features, y_train, val_features, y_val):
    modelSVM = SVC(kernel = "rbf", gamma = 'auto',probability=True, class_weight = "balanced")
    modelSVM.fit(train_features, y_train)
    predictSVM = modelSVM.predict(val_features)

    accuracys=[]

    accuracy = accuracy_score(y_val, predictSVM)
    print("Accuracy SVM = %.3f" % accuracy)

    return predictSVM, accuracys


from sklearn.preprocessing import StandardScaler
train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()

mean_int, std_int = normalizeIntensity(train_slices, train_slices_masks)

train_slices = (train_slices - mean_int)/std_int
val_slices = (val_slices - mean_int)/std_int
test_slices = (test_slices - mean_int)/std_int

train_int_features, val_int_features, test_int_features = getIntensityFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)
train_circ_features, val_circ_features, test_circ_features = getCircularFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)
#train_shape_features, val_shape_features, test_shape_features = getShapeFeatures(train_slices_masks, val_slices_masks,test_slices_masks)
#train_lbp_features, val_lbp_features, test_lbp_features = getLBPFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)
#train_gabor_features, val_gabor_features, test_gabor_features = GetGaborFilter(train_slices, val_slices, test_slices)


print("Intensity Features only \n=======================")
int_train_scaled = StandardScaler().fit_transform(train_int_features)
int_val_scaled = StandardScaler().fit_transform(val_int_features)

prediction_int = getPrediction(int_train_scaled , y_train, int_val_scaled , y_val)
pca = PCA(n_components=2)
pca_train = pca.fit_transform(train_int_features)



plt.plot(pca_train[y_train==0,0], pca_train[y_train==0,1],'ro')
plt.plot(pca_train[y_train==1,0], pca_train[y_train==1,1], 'bo')
plt.plot(pca_train[y_train==2,0], pca_train[y_train==2,1],'co')
plt.show()

plt.plot(train_int_features[y_train==0,2],'ro')
plt.plot(train_int_features[y_train==1,2], 'bo')
plt.plot(train_int_features[y_train==2,2],'co')
plt.show()


print("Circular Features only \n=======================")
circ_train_scaled = StandardScaler().fit_transform(train_circ_features)
circ_val_scaled = StandardScaler().fit_transform(val_circ_features)

prediction_circ = getPrediction(circ_train_scaled , y_train, circ_val_scaled , y_val)
#pca = PCA(n_components=2)
#pca_train = pca.fit_transform(train_circ_features)

"""
plt.plot(pca_train[y_train==0,0], pca_train[y_train==0,1],'ro')
plt.plot(pca_train[y_train==1,0], pca_train[y_train==1,1], 'bo')
plt.plot(pca_train[y_train==2,0], pca_train[y_train==2,1],'co')
plt.show()
"""
plt.plot(train_circ_features[y_train==0,0], train_circ_features[y_train==0,2],'ro')
plt.plot(train_circ_features[y_train==1,0], train_circ_features[y_train==1,2],'bo')
plt.plot(train_circ_features[y_train==2,0],train_circ_features[y_train==2,2],'co')
plt.show()

"""
print("Shape Features only \n=======================")
int_train_scaled = StandardScaler().fit_transform(train_shape_features)
int_val_scaled = StandardScaler().fit_transform(val_shape_features)
prediction_shape = getPrediction(int_train_scaled , y_train, int_val_scaled , y_val)
#prediction_shape = getPrediction(train_shape_features, y_train, val_shape_features, y_val)

pca = PCA(n_components=2)
pca_train = pca.fit_transform(train_shape_features)
plt.plot(pca_train[y_train==0,0], pca_train[y_train==0,1],'ro')
plt.plot(pca_train[y_train==1,0], pca_train[y_train==1,1], 'bo')
plt.plot(pca_train[y_train==2,0], pca_train[y_train==2,1],'co')
plt.show()

"""
"""
print("Gabor Features only \n=======================")
prediction_gb = getPrediction(train_gabor_features, y_train, val_gabor_features, y_val)


pca = PCA(n_components=2)
pca_train = pca.fit_transform(train_gabor_features)
plt.plot(pca_train[y_train==0,0], pca_train[y_train==0,1],'ro')
plt.plot(pca_train[y_train==1,0], pca_train[y_train==1,1], 'bo')
plt.plot(pca_train[y_train==2,0], pca_train[y_train==2,1],'co')
plt.show()


print("LBP Features only \n=======================")
prediction_lbp = getPrediction(train_lbp_features, y_train, val_lbp_features, y_val)
pca = PCA(n_components=2)
pca_train = pca.fit_transform(train_lbp_features)
pca_val = pca.fit_transform(val_lbp_features)
plt.plot(pca_train[y_train==0,0], pca_train[y_train==0,1],'ro')
plt.plot(pca_train[y_train==1,0], pca_train[y_train==1,1], 'bo')
plt.plot(pca_train[y_train==2,0], pca_train[y_train==2,1],'co')
prediction_lbp = getPrediction(pca_train, y_train, pca_val, y_val)
plt.show()

print("All Features\n=======================")
train_features = np.concatenate((train_int_features, train_shape_features,  train_lbp_features), axis=1)
val_features = np.concatenate((val_int_features, val_shape_features, val_lbp_features), axis=1)
test_features = np.concatenate((test_int_features, test_shape_features,  test_lbp_features), axis=1)
prediction_all = getPrediction(train_features, y_train, val_features, y_val)

#train_features = np.concatenate((train_int_features, train_shape_features), axis=1)
#val_features = np.concatenate((val_int_features, val_shape_features), axis=1)
#test_features = np.concatenate((test_int_features, test_shape_features), axis=1)
#prediction_all = getPrediction(train_features, y_train, val_features, y_val)

pca = PCA(n_components=2)
pca_train = pca.fit_transform(train_features)
plt.plot(pca_train[y_train==0,0], pca_train[y_train==0,1],'ro')
plt.plot(pca_train[y_train==1,0], pca_train[y_train==1,1], 'bo')
plt.plot(pca_train[y_train==2,0], pca_train[y_train==2,1],'co')
plt.show()



print("PCA\n=============")
pca = PCA(n_components=10)
pca_train = pca.fit_transform(train_features)
modelSVM = SVC(gamma = 'auto', decision_function_shape='ovo', class_weight='balanced')
modelSVM.fit(pca_train, y_train)
pca_val = pca.fit_transform(val_features)
predictSVM = modelSVM.predict(pca_val)
accuracy = accuracy_score(y_val, predictSVM)

plt.plot(pca_train[y_train==0,0],pca_train[y_train==0,1], 'ro')
plt.plot(pca_train[y_train==1,0],pca_train[y_train==1,1], 'bo')
plt.plot(pca_train[y_train==2,0],pca_train[y_train==2,1], 'co')
plt.show()

print("Accuracy SVM = %.3f" % accuracy)


plt.plot(train_shape_features[y_train==0,0], 'ro')
plt.plot(train_shape_features[y_train==1,0], 'bo')
plt.plot(train_shape_features[y_train==2,0], 'co')
plt.show()
"""
    