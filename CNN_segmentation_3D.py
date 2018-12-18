# -*- coding: utf-8 -*-

from get_data import getData
import numpy as np
import matplotlib.pyplot as plt
from lung_mask import getLungMask

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D, GlobalMaxPool3D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
#import keras

import cv2


#%%


def run_segmentation_CNN():
 
    train_volumes, train_volumes_masks, _, val_volumes, val_volumes_masks, _, test_volumes, test_volumes_masks, _ = getData(type_ = "volume")
        
    train, test, val, trainMasks, testMasks, valMasks=prepare_CNN(train_volumes, train_volumes_masks, val_volumes, val_volumes_masks, test_volumes, test_volumes_masks)
    results, accuracy, dice, jaccard, preds_test_nodules, accuracy_val, dice_val, jaccard_val, preds_val_nodules=train_model(train, test, val, trainMasks, testMasks, valMasks)
    plot_loss(results)
    print("Test set: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice, jaccard, accuracy))
    print("validation set: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice_val, jaccard_val, accuracy_val))
    
 
#%%
    """
prepare_CNN
===============
prepares the input for the model of the CNN

Arguments:
    
Returns:train_volumes, test_volumes, val_volumes - images of train, test and validation sets after aplying lung mask, normalization and 
reshaped for input on the CNN
     train_volumes_masks, test_volumes_masks, val_volumes_masks  - classes in the format of one-hot-vector (1,0,0)
"""

def prepare_CNN(train_volumes, train_volumes_masks, val_volumes, val_volumes_masks, test_volumes, test_volumes_masks):
    
    #Aplly lung mask
    for i in range(len(train_volumes)):
        chull=getLungMask(train_volumes[i])
        train_volumes[i][chull == 0] = 0
    for i in range(len(val_volumes)):
        chull=getLungMask(val_volumes[i])
        val_volumes[i][chull == 0] = 0
    for i in range(len(test_volumes)):
        chull=getLungMask(test_volumes[i])
        test_volumes[i][chull == 0] = 0
        
        

    mean_int=np.mean(train_volumes)
    std_int=np.std(train_volumes)
    
    train_volumes = (train_volumes - mean_int)/std_int
    val_volumes = (val_volumes - mean_int)/std_int
    test_volumes = (test_volumes - mean_int)/std_int
    
    
    #reshape to a multiple of 16 to better applye the U-net CNN - padding from 51 to 64
    train_volumes= [cv2.copyMakeBorder(train_volumes[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(train_volumes))]
    val_volumes= [cv2.copyMakeBorder(val_volumes[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(val_volumes))]
    test_volumes= [cv2.copyMakeBorder(test_volumes[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(test_volumes))]
    train_volumes_masks= [cv2.copyMakeBorder(train_volumes_masks[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(train_volumes_masks))]
    val_volumes_masks= [cv2.copyMakeBorder(val_volumes_masks[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(val_volumes_masks))]


    train_volumes_masks = np.asarray(train_volumes_masks)
    test_volumes_masks = np.asarray(test_volumes_masks)
    val_volumes_masks = np.asarray(val_volumes_masks)
    
    train_volumes = np.asarray(train_volumes)
    test_volumes = np.asarray(test_volumes)
    val_volumes = np.asarray(val_volumes)
    
    train_volumes_masks = train_volumes_masks.astype('float32')
    test_volumes_masks = test_volumes_masks.astype('float32')
    val_volumes_masks = val_volumes_masks.astype('float32')
   
    train_volumes = train_volumes.astype('float32')
    test_volumes = test_volumes.astype('float32')
    val_volumes = val_volumes.astype('float32')
   
    train_volumes = train_volumes.reshape(-1,64,64,1)
    test_volumes = test_volumes.reshape(-1,64,64,1)
    val_volumes = val_volumes.reshape(-1, 64,64, 1)
    
    train_volumes_masks = train_volumes_masks.reshape(-1,64,64,1)
    
    val_volumes_masks = val_volumes_masks.reshape(-1, 64,64, 1)
    
   
    

    return train_volumes, test_volumes, val_volumes, train_volumes_masks, test_volumes_masks, val_volumes_masks

#%%

def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size,kernel_size ), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size,kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x



def get_unet(input_img, n_filters=16, dropout=0.4, batchnorm=True):
    
    
    # contracting path
    c1 = conv3d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling3D((2, 2,2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv3d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling3D((2, 2,2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv3d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling3D((2, 2,2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv3d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling3D(pool_size=(2, 2,2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv3d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv3DTranspose(n_filters*8, (3, 3,3), strides=(2, 2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv3d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv3DTranspose(n_filters*4, (3, 3, 3), strides=(2, 2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv3d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv3DTranspose(n_filters*2, (3, 3, 3), strides=(2, 2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv3d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv3DTranspose(n_filters*1, (3, 3, 3), strides=(2, 2,2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv3d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%%
    """
IoU_loss
===============
defenition of loss for binary problem - try to maximize the jaccard coefficient ( as only true values matter)
it solves the problem of having more false (0) pixeis

Arguments:
    
Returns:
    * results- coefiicient to minimize (1-jaccard)
"""

from keras import backend as K

def IoU_loss(y_true,y_pred):
    smooth = 1e-12
    # author = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(1-jac)

#%%
"""
train_model
===============
train the model with tarin set and validation set to define treshold - evaluates test set

Arguments:
    
Returns:
    * results- result of the trained model with keras
    accuracy, dice, jaccard - evaluation scores for the test set
    preds_test_nodules - predicted nodules on test set
"""

def train_model(train_volumes, test_volumes, val_volumes, train_volumes_masks, test_volumes_masks, val_volumes_masks):
    # define parameters
    im_width = 64
    im_height = 64
    epochs=100
    batch=len(train_volumes)
    
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=3, dropout=0.05, batchnorm=True)
    
    
    model.compile(optimizer=Adam(), loss=IoU_loss)
    #model.summary()
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model3dsegmentação.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    
    results = model.fit(train_volumes, train_volumes_masks, batch_size=batch,steps_per_epoch=10, epochs=epochs, callback=callbacks, verbose=0, validation_data=(val_volumes, val_volumes_masks))
    
    model.load_weights('model3dsegmentação.h5')
    
    
    treshold=(0.35,0.4, 0.45, 0.5,0.55,0.6,0.65,0.7,0.75)
    maximo=0
    # Predict for test with treshold
    preds_train = model.predict(train_volumes, verbose=0)
    for tresh in treshold:
        
        preds_train_nodules = (preds_train >tresh).astype(np.uint8)
        
        preds_train_nodules=preds_train_nodules.reshape(-1,64,64)
        train_volumes_masks=train_volumes_masks.reshape(-1,64,64)
        _, dice, jaccard = confusionMatrix(np.hstack(np.hstack(preds_train_nodules)), np.hstack(np.hstack(train_volumes_masks)))
        
        metrics=dice+jaccard  # the best result will dictate which is the bst treshold
        
        if metrics > maximo :
            maximo=metrics
            best_treshold=tresh
            
    # Predict for test with treshold already defined by training set
    preds_val = model.predict(val_volumes, verbose=0)
    preds_val_nodules = (preds_val >best_treshold).astype(np.uint8)
    
    val_volumes_masks=val_volumes_masks.reshape(-1,64,64)
    preds_val_nodules=preds_val_nodules.reshape(-1,64,64)
    
    
    accuracy_val, dice_val, jaccard_val = confusionMatrix(np.hstack(np.hstack(preds_val_nodules)), np.hstack(np.hstack(val_volumes_masks)))

    # Predict for test with treshold already defined by training set
    preds_test = model.predict(test_volumes, verbose=0)
    preds_test_nodules = (preds_test >best_treshold).astype(np.uint8)
    
    preds_test_nodules=preds_test_nodules.reshape(-1,64,64)
    #test_volumes_masks=test_volumes_masks.reshape(-1,51,51)
    
    #cut the border previously used to match the ground truth
    border_size_top_right=6
    border_size_bottom_left=6
    preds_test_nodules=[nodule[border_size_top_right:-(border_size_top_right+1),border_size_bottom_left:-(border_size_bottom_left+1)] for nodule in preds_test_nodules]
    
    #Aplly morphologic operation to close some holes on predicted images
    preds_test_nodules=closing(preds_test_nodules)
    
    
    accuracy, dice, jaccard = confusionMatrix(np.hstack(np.hstack(preds_test_nodules)), np.hstack(np.hstack(test_volumes_masks)))

    return results, accuracy, dice, jaccard, preds_test_nodules, accuracy_val, dice_val, jaccard_val, preds_val_nodules


#%%
"""
closing- morpological closing operation
=================================================
Arguments: image array
return: image array after closing
"""
def closing(preds_image):
    new_preds=[]
    for i in range(len(preds_image)):
       
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dilated_mask = cv2.dilate(preds_image[i],kernel_ellipse,iterations = 2)
        erode_mask = cv2.erode(dilated_mask,kernel_ellipse,iterations = 2) 
        
        new_preds.append(erode_mask)
    return new_preds
       
#%%

def confusionMatrix(predictions, labels):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    predictions= predictions.astype('float32')
    labels = labels.astype('float32')
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

    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    
    dice = (2*true_positives/(false_positives+false_negatives+(2*true_positives)))
    jaccard = (true_positives)/(true_positives+false_positives+false_negatives)
                
    return accuracy, dice, jaccard


#%%
"""
show loss
===============
shows the progression of loss during the training of the model

Arguments: results - model trained
    
Returns:
    *void
"""
def plot_loss(results):
    
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], 'bo', label="loss")
    plt.plot(results.history["val_loss"],'b', label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();


#%%
run_segmentation_CNN()
