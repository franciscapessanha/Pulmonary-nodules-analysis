import numpy as np
from skimage.filters.rank import entropy
import cv2 as cv
from skimage.measure import label, regionprops

"""
Get Intensity Features
==============

Calculates the intensity and circularity features.

Arguments:
    * train_slices
    * train_slices_masks
    * val_slices
    * val_slices_masks
    
Return:
    * train_int: intensity for the training set
    * val_int: intensity for the validation or test set
    * train_circ: circularty for the training set
    * val_circ: circularty for the validation set

"""

def getIntensityFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks):
    train_int = np.vstack(calcIntensityFeatures(train_slices, train_slices_masks))
    val_int = np.vstack(calcIntensityFeatures(val_slices, val_slices_masks))
    
    train_circ = np.vstack(calcCircularFeatures(train_slices, train_slices_masks))
    val_circ = np.vstack(calcCircularFeatures(val_slices, val_slices_masks))
 
    return train_int, val_int, train_circ, val_circ


"""
Calculate Circularity Features
==============

Calculates the centroid of the circle and creates a circle with different radius in order to avaliate different parameters within the nodule. 

Arguments:
    * nodules
    * masks
    
Return:
    * circular_features: mean and standard deviation of the circular features

"""

def calcCircularFeatures(nodules, masks):
    circular_features = []
    for nodule, mask in zip(nodules, masks):
        circle_0 = np.zeros(np.shape(nodule), np.uint8)
        circle_1 = np.zeros(np.shape(nodule), np.uint8)
        circle_2 = np.zeros(np.shape(nodule), np.uint8)
       
        region = label(mask)
        props = regionprops(region)
        minor_axis = props[0]['minor_axis_length']
        centroid = props[0]['centroid']

        cv.circle(circle_0, (int(centroid[0]), int(centroid[1])),int(minor_axis*0.05),1, -1)
        cv.circle(circle_1, (int(centroid[0]), int(centroid[1])),int(minor_axis*0.30),1, -1)
        cv.circle(circle_2, (int(centroid[0]), int(centroid[1])),int(minor_axis*0.55),1, -1)

        
        mean_0 = np.mean(nodule[circle_0 == 1])
        mean_1 = np.mean(nodule[circle_1 == 1])
        mean_2 = np.mean(nodule[circle_2 == 1])
        
        std_0 = np.mean(nodule[circle_0 == 1])
        std_1 = np.mean(nodule[circle_1 == 1])
        std_2 = np.mean(nodule[circle_2 == 1])

        
        circular_features.append([mean_0, std_0, mean_1, std_1, mean_2, std_2])

    return circular_features

"""
Calculate Intensity Features
==============

Calculates the mean, stardard deviation, maximum and minimun of the nodule and creates an intensity array

Arguments:
    * nodules
    * masks
    
Return:
    * intensity_features: intensity of the nodule

"""

def calcIntensityFeatures(nodules, masks):
    intensity_features = []
    for nodule, mask in zip(nodules, masks):
        mean = np.mean(nodule[mask == 1])
        std = np.std(nodule[mask == 1])
        max_ = np.max(nodule[mask == 1])
        min_ = np.min(nodule[mask == 1])
        intensity_features.append([mean, std, max_, min_])  
    
    return intensity_features
    
