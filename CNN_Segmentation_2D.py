
from get_data import getData
import numpy as np
import matplotlib.pyplot as plt
from lung_mask import getLungMask

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#import keras
#train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()
#plt.imshow(test_slices_masks[1], cmap=plt.cm.gray)
import cv2

#%%


def run_segmentation_CNN():
    train_slices, train_slices_masks, y_train, val_slices, val_slices_masks, y_val, test_slices, test_slices_masks, y_test = getData(mode="cross_val")
    
    for train_x, train_masks, val_x, val_masks in zip(train_slices, train_slices_masks, val_slices, val_slices_masks):
        
        train, test, val, trainMasks, testMasks, valMasks=prepare_CNN(train_x, train_masks, val_x, val_masks, test_slices, test_slices_masks)
        results, accuracy, dice, jaccard, preds_test_nodules, accuracy_val, dice_val, jaccard_val, preds_val_nodules=train_model(train, test, val, trainMasks, testMasks, valMasks)
        plot_loss(results)
        print("Test set: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice, jaccard, accuracy))
        print("validation set: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice_val, jaccard_val, accuracy_val))

    return preds_test_nodules, preds_val_nodules

#%%
    """
prepare_CNN
===============
prepares the input for the model of the CNN

Arguments:
    
Returns:train_slices, test_slices, val_slices - images of train, test and validation sets after aplying lung mask, normalization and 
reshaped for input on the CNN
     train_slices_masks, test_slices_masks, val_slices_masks  - classes in the format of one-hot-vector (1,0,0)
"""

def prepare_CNN(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks):
    
    #Aplly lung mask
    for i in range(len(train_slices)):
        chull=getLungMask(train_slices[i])
        train_slices[i][chull == 0] = 0
    for i in range(len(val_slices)):
        chull=getLungMask(val_slices[i])
        val_slices[i][chull == 0] = 0
    for i in range(len(test_slices)):
        chull=getLungMask(test_slices[i])
        test_slices[i][chull == 0] = 0
        
        

    mean_int=np.mean(train_slices)
    std_int=np.std(train_slices)
    
    train_slices = (train_slices - mean_int)/std_int
    val_slices = (val_slices - mean_int)/std_int
    test_slices = (test_slices - mean_int)/std_int
    
    
    #reshape to a multiple of 16 to better applye the U-net CNN - padding from 51 to 64
    train_slices= [cv2.copyMakeBorder(train_slices[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(train_slices))]
    val_slices= [cv2.copyMakeBorder(val_slices[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(val_slices))]
    test_slices= [cv2.copyMakeBorder(test_slices[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(test_slices))]
    train_slices_masks= [cv2.copyMakeBorder(train_slices_masks[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(train_slices_masks))]
    test_slices_masks= [cv2.copyMakeBorder(test_slices_masks[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(test_slices_masks))]
    val_slices_masks= [cv2.copyMakeBorder(val_slices_masks[i],7,6,6,7,cv2.BORDER_CONSTANT,value=0) for i in range(len(val_slices_masks))]
    

    train_slices_masks = np.asarray(train_slices_masks)
    test_slices_masks = np.asarray(test_slices_masks)
    val_slices_masks = np.asarray(val_slices_masks)
    
    train_slices = np.asarray(train_slices)
    test_slices = np.asarray(test_slices)
    val_slices = np.asarray(val_slices)
    
    train_slices_masks = train_slices_masks.astype('float32')
    test_slices_masks = test_slices_masks.astype('float32')
    val_slices_masks = val_slices_masks.astype('float32')
   
    train_slices = train_slices.astype('float32')
    test_slices = test_slices.astype('float32')
    val_slices = val_slices.astype('float32')
   
    train_slices = train_slices.reshape(-1,64,64,1)
    test_slices = test_slices.reshape(-1,64,64,1)
    val_slices = val_slices.reshape(-1, 64,64, 1)
    
    train_slices_masks = train_slices_masks.reshape(-1,64,64,1)
    test_slices_masks = test_slices_masks.reshape(-1,64,64,1)
    val_slices_masks = val_slices_masks.reshape(-1, 64,64, 1)
    
   
    

    return train_slices, test_slices, val_slices, train_slices_masks, test_slices_masks, val_slices_masks

#%%

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x



def get_unet(input_img, n_filters=16, dropout=0.4, batchnorm=True):
    
    
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

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

def train_model(train_slices, test_slices, val_slices, train_slices_masks, test_slices_masks, val_slices_masks):
    # define parameters
    im_width = 64
    im_height = 64
    epochs=100
    batch=len(train_slices)
    
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=3, dropout=0.05, batchnorm=True)
    
    model.compile(optimizer=Adam(), loss="binary_crossentropy")
    #model.summary()
    
    results = model.fit(train_slices, train_slices_masks, batch_size=batch, epochs=epochs, verbose=0, validation_data=(val_slices, val_slices_masks))
    
    treshold=(0.35,0.4, 0.45, 0.5,0.55,0.6,0.65,0.7,0.75)
    maximo=0
    # Predict for test with treshold
    preds_train = model.predict(train_slices, verbose=0)
    for tresh in treshold:
        
        preds_train_nodules = (preds_train >tresh).astype(np.uint8)
        
        preds_train_nodules=preds_train_nodules.reshape(-1,64,64)
        train_slices_masks=train_slices_masks.reshape(-1,64,64)
        _, dice, jaccard = confusionMatrix(np.hstack(np.hstack(preds_train_nodules)), np.hstack(np.hstack(train_slices_masks)))
        
        metrics=dice+jaccard  # the best result will dictate which is the bst treshold
        
        if metrics > maximo :
            maximo=metrics
            best_treshold=tresh
            
    # Predict for test with treshold already defined by training set
    preds_val = model.predict(val_slices, verbose=0)
    preds_val_nodules = (preds_val >best_treshold).astype(np.uint8)
    
    val_slices_masks=val_slices_masks.reshape(-1,64,64)
    preds_val_nodules=preds_val_nodules.reshape(-1,64,64)
    
    
    accuracy_val, dice_val, jaccard_val = confusionMatrix(np.hstack(np.hstack(preds_val_nodules)), np.hstack(np.hstack(val_slices_masks)))

    # Predict for test with treshold already defined by training set
    preds_test = model.predict(test_slices, verbose=0)
    preds_test_nodules = (preds_test >best_treshold).astype(np.uint8)
    
    preds_test_nodules=preds_test_nodules.reshape(-1,64,64)
    test_slices_masks=test_slices_masks.reshape(-1,64,64)
    
    accuracy, dice, jaccard = confusionMatrix(np.hstack(np.hstack(preds_test_nodules)), np.hstack(np.hstack(test_slices_masks)))

    return results, accuracy, dice, jaccard, preds_test_nodules, accuracy_val, dice_val, jaccard_val, preds_val_nodules

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
def plot_loss(results):
    
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();

#%%
    
preds_test_nodules, preds_val_nodules=run_segmentation_CNN()