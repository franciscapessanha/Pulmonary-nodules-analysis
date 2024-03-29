# -*- coding: utf-8 -*-
from get_data import getData
import numpy as np
from lung_mask import getLungMask
from matplotlib import pyplot as plt
from hessian_3D import getEigNodules, gaussianSmooth, getSI, getCV, getVmed
from sklearn.svm import SVC
from skimage.measure import label, regionprops
from sklearn.neighbors import KNeighborsClassifier
"""
Run
===============================================================================
Allows to run the code for 3D segmentation, choosing default or cross-validation mode. 

Arguments:
    * mode: default or cross-validation
    
Return:
    * void
"""

def run(mode = "default"):
    if mode == "default": 
        train_volumes, train_masks, _, val_volumes, val_masks, _, test_volumes, test_masks, _ = getData(mode = "default", type_ = "volume")
        get3DSegmentation(train_volumes, train_masks, val_volumes, val_masks, test_volumes, test_masks)
        
    elif mode == "cross_val":
    
        cv_train_volumes, cv_train_masks, _ , cv_val_volumes, cv_val_masks, _, test_volumes, test_masks, _ = getData(mode = "cross_val", type_ = "volume")
        for train_volumes, train_masks, val_volumes, val_masks in zip(cv_train_volumes, cv_train_masks, cv_val_volumes, cv_val_masks):
            get3DSegmentation(train_volumes, train_masks, val_volumes, val_masks, test_volumes, test_masks)
            
"""
Get 3D Segmentation
===============================================================================

Gets images, labels, mean and standard deviation from the training set and the validation/test set. Fits models (SVM and kNN) with training set and predicts the labels for new sets..

Arguments:
    * train_x: train set
    * train_masks: train masks set (ground truth)
    * val_x: validation set
    * val_masks: validation masks set (ground truth)
    * test_x: test set
    * test_masks: test masks set (ground truth)
    
Return:
    * void

"""
def get3DSegmentation(train_volumes, train_masks, val_volumes, val_masks, test_volumes, test_masks):
    
    print("1. Get Training set")
    points, labels, mean_int, std_int = getTrainingSet(train_volumes, train_masks, 0.10 )
    print("2. Get Input set")
    val_lung, labels_val_lung  = getInputSet(val_volumes, val_masks, mean_int, std_int)

    #SVM
    model_SVM = SVC(kernel = 'rbf', random_state = 1,gamma='auto')
    model_SVM.fit(points,labels)
    
    #kNn
    model_kNN = KNeighborsClassifier(n_neighbors=7)
    model_kNN.fit(points,labels)
    
    pred_val_lung = []
    result_val = []
    for x, sample in zip(val_lung, val_volumes):
        pred_lung = model_SVM.predict(np.transpose(np.vstack(x)))
        pred_val_lung.append(pred_lung) 
        result_val.append(np.hstack(showResults(pred_lung, sample, "volume")))
    
    predictions_outer_lung, labels_outer_lung = outerLungPrediction(val_volumes, val_masks)
    dice, jaccard, matrix, accuracy  = getPerformanceMetrics(np.hstack(result_val), np.hstack(labels_val_lung), predictions_outer_lung, labels_outer_lung)
    print("3. SVM val \n=======")
    print("The dice value is %.2f and the jaccard value is %.2f" % (dice, jaccard))

    print("4. SVM test \n=======")
    test_lung, labels_test_lung  = getInputSet(test_volumes, test_masks, mean_int, std_int)
    
    pred_test_lung = []
    result_test = []
    for x, sample in zip(test_lung, test_volumes):
        pred_lung = model_SVM.predict(np.transpose(np.vstack(x)))
        pred_test_lung.append(pred_lung) 
        result_test.append(np.hstack(showResults(pred_lung, sample, "volume")))

    predictions_outer_lung, labels_outer_lung = outerLungPrediction(test_volumes, test_masks)
    dice, jaccard, matrix, accuracy  = getPerformanceMetrics(np.hstack(result_test), np.hstack(labels_test_lung), predictions_outer_lung, labels_outer_lung)
    print("The dice value is %.2f and the jaccard value is %.2f" % (dice, jaccard))

    print("5. kNN test \n=======")
    pred_test_lung_knn = []
    result_test_knn = []
    for x, sample, mask in zip(test_lung, test_volumes):
        pred_lung_knn = model_kNN.predict(np.transpose(np.vstack(x)))
        pred_test_lung_knn.append(pred_lung_knn) 
        result_test_knn.append(np.hstack(showResults(pred_lung_knn, sample, "volume")))

    predictions_outer_lung, labels_outer_lung = outerLungPrediction(test_volumes, test_masks)
    dice, jaccard, matrix, accuracy = getPerformanceMetrics(np.hstack(result_test_knn), np.hstack(labels_test_lung), predictions_outer_lung, labels_outer_lung)
    print("The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice, jaccard))

"""
Process results 
===============================================================================
"""   
def processResult(prediction, sample, lung_mask):
  
    label_image = label(prediction)
    props = regionprops(label_image)
    areas = [r.area for r in props]
    areas.sort()
        
    for l in range(1,np.max(label_image)): 
        if props[l-1]['area'] < 0.75 * areas[-1]:
            prediction[label_image == l] = 0
    
     #plt.imshow(result, cmap = "gray")
     #plt.show()  
     #plt.imshow(sample, cmap = "gray")
     #plt.show()
        
    return prediction
"""
Show results 
===============================================================================
Prints the original nodule image, the corresponding ground truth image and the prediction obtained previously

Arguments:
    * prediction_lung: segmented nodule
    * sample: original nodule image 
    * mask: ground truth

    
Return:
    * nodule segmention result

 """            
def showResults(prediction_lung, sample, type_):
    
    if type_ == "slice":
        lung_mask = getLungMask(sample)
        prediction = np.zeros((51,51,51), np.uint8)
        prediction[lung_mask == 1] = prediction_lung
        result = processResult(prediction, sample, lung_mask)
        return result[lung_mask == 1]
    
    if type_ == "volume":
        result = np.zeros((51,51,51), np.uint8)
        prediction = np.zeros((51,51,51), np.uint8)
        lung_mask = getLungMask(sample, "volume")
        prediction[lung_mask == 1] = prediction_lung
        for slice_ in range(len(sample)):
            result[slice_,:,:] = processResult(prediction[slice_,:,:],slice_, lung_mask[slice_,:,:])
        
        return result[lung_mask == 1]
"""
Separate Features
==============================================================================
Separates the sample features according to the ground truth binary classification: 
nodule and non-nodule.

Arguments:
    * sample_features: features for each point of the samples
    * sample
    * mask
Returns:
    * features_n: numpy array will the features corresponding to the nodule points
    * features_nn: numpy array will the features corresponding to the non nodule points,
    excluding the outer lung points
"""

def separateFeatures(sample_features, sample, mask):
   features_n = []
   features_nn = [] 
   for feature in sample_features:
       features_n.append(feature[mask == 1])
       lung_mask= getLungMask(sample, type_ = "volume") - mask
       features_nn.append(feature[lung_mask==1])
    
   features_n = np.asarray(features_n)
   features_nn = np.asarray(features_nn)
   return features_n, features_nn

"""
Get Indexes
==============================================================================
Returns the indexes corresponding to the non-nodule and nodule features points.

Arguments:
    * features_n: numpy array will the features corresponding to the nodule points
    * features_nn: numpy array will the features corresponding to the non nodule points,
    excluding the outer lung points
Returns:
    * i_nodules: numpy array with the indexes for the nodule features points
    * i_non_nodules: numpy array with the indexes for the non nodule features points
"""

def getIndexes(features_n, features_nn):
    #ir buscar os indices de cada pixel
    i_nodules = np.asarray([j for j in range(len(features_n[0]))])
    i_non_nodules = np.asarray([j for j in range(len(features_nn[0]))])
    np.random.shuffle(i_nodules)
    np.random.shuffle(i_non_nodules)
    return i_nodules, i_non_nodules

"""
Get Feature Points
==============================================================================
Returns a list with the training points and corresponding labels.

Arguments:
    * features_n: numpy array will the features corresponding to the nodule points
    * features_nn: numpy array will the features corresponding to the non nodule points,
    excluding the outer lung points
    * number_of_pixel: number of points to extract, per image, of each kind of nodule
        if <= 1: the number of pixels is in percentage
        else: is an absolute value
Returns:
    * points: points extracted from the features_n and features_nn according to 
    the number_of_pixel
    * labels: points labels (nodule = 1, non_nodule = 0)
"""
      
def getFeaturePoints(features_n, features_nn, number_of_pixel):
    i_nodules, i_non_nodules = getIndexes(features_n,features_nn)

    if number_of_pixel <= 1: #percentage
        i_n = [i_nodules[j] for j in range(int(len(i_nodules) * number_of_pixel))]
        i_nn = [i_non_nodules[j] for j in range(int(len(i_nodules) * number_of_pixel))]
    
    else:
        i_n = [i_nodules[j] for j in range(number_of_pixel)]
        i_nn = [i_non_nodules[j] for j in range(number_of_pixel)]
    
    nodule_points = [features_n[:,i] for i in i_n]
    non_nodule_points = [features_nn[:,i] for i in i_nn]
    
    points = np.concatenate((nodule_points, non_nodule_points), axis = 0)
    labels = np.concatenate((np.ones(np.shape(nodule_points)[0]),np.zeros(np.shape(non_nodule_points)[0])), axis = 0)
    return points, labels

"""
Normalize Images
==============================================================================
Calculates the mean and std of the training set in order to normalize all the 
samples.

Arguments:
    * train_images
Returns:
    * mean: mean intensity of the train_images
    * std: std of the train_images
"""

def normalizeImages(train_images):
    all_px = []
    for nodule in train_images:
        all_px.append(nodule)
    all_px = np.hstack(all_px)
    mean = np.mean(all_px)
    std = np.std(all_px)
    return mean, std

"""
Get Training Set
==============================================================================
Normalizes the training set, applies gaussian filters, calculates the nodules eigenvalues and SI, CV and Vmed 
Concatenates all features

Arguments:
    * train_slices: images from the training set
    * train_slices_masks: masks (ground truth images) from the training set
    * number_of_pixel: number of pixels that we want to get from the mask 
    
Returns:
    * points: nodules pixels from training set
    * labels: pixels labels from training set
    * mean_int: mean from the normalized training set images 
    * std_int: standard deviation from the normalized training set images
"""

def getTrainingSet(train_volumes, train_masks, number_of_pixel):
    mean_int, std_int = normalizeImages(train_volumes)
    norm_volumes = [(nodule - mean_int)/std_int for nodule in train_volumes]
    
    smooth_img = gaussianSmooth(norm_volumes)
    eigen_nodules = getEigNodules(smooth_img)
    SI_nodules = getSI (eigen_nodules)
    CV_nodules = getCV(eigen_nodules)
    Vmed_nodules = getVmed(eigen_nodules)
    
    for i in range(len(train_volumes)):
        sample_features = [norm_volumes[i], SI_nodules[i], CV_nodules[i], Vmed_nodules[i]]
        sample = train_volumes[i]
        mask = train_masks[i]
    
        features_n, features_nn = separateFeatures(sample_features, sample, mask)
    
        if i == 0:
            points, labels = getFeaturePoints(features_n, features_nn, number_of_pixel)
        else:
            p, l = getFeaturePoints(features_n, features_nn, number_of_pixel)
            points = np.concatenate((points, p), axis = 0)
            labels = np.concatenate((labels, l), axis = 0)
            
    return points, labels, mean_int, std_int

"""
Get Input Set
==============================================================================
Normalizes the validation or test set, applies gaussian filters, calculates the nodules eigenvalues and SI, CV and Vmed 
Concatenates all features

Arguments:
    * nodules: images from the input set (validation or test set)
    * masks: masks (ground truth images) from the input set (validation or test set)
    * mean_int: mean from the normalized training set images 
    * std_int: standard deviation from the normalized training set images
    
Returns:
    * input_set: pixels nodules, si, cv and vmed values per pixel
    * mask_px: mask that results from the application of the lung mask

"""
def getInputSet(nodules, masks,mean_int, std_int):
    norm_nodules = [(nodule - mean_int)/std_int for nodule in nodules]
    smooth_img = gaussianSmooth(norm_nodules)
    eigen_nodules = getEigNodules(smooth_img)
    SI_nodules = getSI(eigen_nodules)
    CV_nodules = getCV(eigen_nodules)
    Vmed_nodules = getVmed(eigen_nodules)
    

    masked_nodules = []
    masked_SI = []
    masked_CV = []
    masked_Vmed = []
    masked_gt = []
    
    nodules_px = []
    si_px = []
    cv_px = []
    Vmed_px = []
    mask_px = []
    
    for nodule, norm_nodule, si, cv_value, v_med, mask in zip(nodules,norm_nodules, SI_nodules, CV_nodules, Vmed_nodules, masks) :
        lung_mask = getLungMask(nodule, type_ = "volume")
        masked_nodules.append(norm_nodule[lung_mask == 1])
        masked_SI.append(si[lung_mask == 1])
        masked_CV.append(cv_value[lung_mask == 1])
        masked_Vmed.append(v_med[lung_mask == 1])
        masked_gt.append(mask[lung_mask == 1])
        
    for i in(range(len(masked_nodules))):
            nodules_px.append(masked_nodules[i])
            si_px.append(masked_SI[i])
            cv_px.append(masked_CV[i])
            Vmed_px.append(masked_Vmed[i])
            mask_px.append(masked_gt[i])

    input_set = np.transpose(np.asarray((nodules_px, si_px, cv_px, Vmed_px)))
    
    return (input_set , mask_px)
"""
Outer Lung Prediction
==============================================================================
Arguments:
    * nodules: nodules images
    * masks: masks (ground truth images) 

Returns:
    * predictions_outer_lung
    * labels_outer_lung
"""
def outerLungPrediction(nodules, masks):
    labels_outer_lung = []
    predictions_outer_lung = []
    for nodule, mask in zip(nodules, masks):
        lung_mask = getLungMask(nodule,  type_ = "volume")
        
        labels_outer_lung.append(mask[lung_mask==0])
        predictions_outer_lung.append(np.zeros(np.shape(nodule[lung_mask==0])))
    
    return np.hstack(predictions_outer_lung), np.hstack(labels_outer_lung)

"""
Confusion Matrix
==============================================================================
Calculates the confusion matrix given a prediction and a label

Arguments:
    * predictions: results obtained from the classifier
    * labels: ground truth of the classification
        
Return:
    * true_positives
    * false_negatives
    * false_positives
    * true_negatives
    
"""   
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

"""
Get Performance Metrics
==============================================================================

Calculates accuracy, precision, recall, auc for evaluation given an array of predictions and the corresponding ground truth

Arguments: 
    * predictions- array with predicted results
    * labels-  corresponding ground true

Return: 
    * accuracy 
    * precision 
    * recall 
    * auc 
"""
def getPerformanceMetrics(predictions_lung, labels_lung, predictions_outer_lung, labels_outer_lung):
    c_matrix_lung = confusionMatrix(predictions_lung, labels_lung)
    c_matrix_outer_lung = confusionMatrix(predictions_outer_lung, labels_outer_lung)
    
    true_positives = c_matrix_lung[0,0] + c_matrix_outer_lung[0,0]
    false_negatives = c_matrix_lung[0,1] + c_matrix_outer_lung[0,1]
    false_positives = c_matrix_lung[1,0] + c_matrix_outer_lung[1,0]
    true_negatives = c_matrix_lung[1,1] + c_matrix_outer_lung[1,1]

    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    
    dice = (2*true_positives/(false_positives+false_negatives+(2*true_positives)))
    jaccard = (true_positives)/(true_positives+false_positives+false_negatives)
    matrix = np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    return dice, jaccard, matrix, accuracy 

run(mode = "cross_val")

