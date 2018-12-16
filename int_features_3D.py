import numpy as np
from skimage.filters.rank import entropy
import cv2 as cv
from skimage.measure import label, regionprops
from scipy.spatial import distance

def getIntensityFeatures(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks):
    train_int = np.vstack(calcIntensityFeatures(train_slices, train_slices_masks))
    val_int = np.vstack(calcIntensityFeatures(val_slices, val_slices_masks))
    test_int = np.vstack(calcIntensityFeatures(test_slices, test_slices_masks))
    
    train_circ = np.vstack(calcCircularFeatures(train_slices, train_slices_masks))
    val_circ = np.vstack(calcCircularFeatures(val_slices, val_slices_masks))
    test_circ = np.vstack(calcCircularFeatures(test_slices, test_slices_masks))
    
    return train_int, val_int, test_int, train_circ, val_circ, test_circ

def calcCircularFeatures(nodules, masks):
    circular_features = []
    for nodule, mask in zip(nodules, masks): 
        
        sum_slices = []
        for slice_ in mask:
            sum_slices.append(np.sum(slice_))
        center_slice = sum_slices.index(np.max(sum_slices))
        
        region = label(mask[center_slice,:,:])
        props = regionprops(region)
        minor_axis = props[0]['minor_axis_length']
        centroid = props[0]['centroid']
        centroid = (int(centroid[0]), int(centroid[1]),center_slice)
        
        dist_0 = int(minor_axis * 0.05)
        dist_1 = int(minor_axis * 0.30)
        dist_2 = int(minor_axis * 0.55)
       
        sphere_0 = np.zeros(np.shape(nodule), np.uint8)
        sphere_1 = np.zeros(np.shape(nodule), np.uint8)
        sphere_2 = np.zeros(np.shape(nodule), np.uint8)
       
        for x in range(len(nodule)):
           for y in range(len(nodule)):
               for z in range(len(nodule)):
                   if distance.euclidean((x,y,z), centroid) <= dist_0:
                       sphere_0[x,y,z] = 1
                   if distance.euclidean((x,y,z), centroid) <= dist_1:
                       sphere_1[x,y,z] = 1
                   if distance.euclidean((x,y,z), centroid) <= dist_2:
                       sphere_2[x,y,z] = 1
        mean_0 = np.mean(nodule[sphere_0 == 1])
        mean_1 = np.mean(nodule[sphere_1 == 1])
        mean_2 = np.mean(nodule[sphere_2 == 1])

        std_0 = np.mean(nodule[sphere_0 == 1])
        std_1 = np.mean(nodule[sphere_1 == 1])
        std_2 = np.mean(nodule[sphere_2 == 1])
        
        circular_features.append([mean_0, std_0, mean_1, std_1, mean_2, std_2])

    return circular_features

def calcIntensityFeatures(nodules, masks):
    intensity_features = []
    for nodule, mask in zip(nodules, masks):
        mean = np.mean(nodule[mask == 1])
        std = np.std(nodule[mask == 1])
        intensity_features.append([mean, std])  
    
    return intensity_features
    
