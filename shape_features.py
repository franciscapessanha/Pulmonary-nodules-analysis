import numpy as np
import cv2 as cv
from skimage.measure import label, regionprops

def getShapeFeatures(train_slices_masks,val_slices_masks, test_slices_masks):
    train_shape_features = np.vstack(calcShapeFeatures(train_slices_masks))
    val_shape_features = np.vstack(calcShapeFeatures(val_slices_masks))
    test_shape_features = np.vstack(calcShapeFeatures(test_slices_masks))
    return train_shape_features, val_shape_features, test_shape_features

def calcShapeFeatures(masks):
    shape_features = []
    for mask in masks:  
        region = label(mask)
        props = regionprops(region)
        area = props[0]['area']
        perimeter = props[0]['perimeter']
        compactness = (perimeter**2)/ (4 * np.pi * area)
        circularity = 4 * np.pi * area / ( perimeter**2 )
        shape_features.append([compactness, circularity])
    return shape_features

