
from get_data import getData
import numpy as np
import matplotlib.pyplot as plt
from lung_mask import getLungMask


from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
#import keras

import cv2


#%%
"""
Runs all code regarding segmentation 2D for CNN aproach

arguments: mode - default rusn 1 time or cross_val runs 5 times with different sets of trains/validation

returns: void
"""


def run_segmentation_CNN(mode = "default"):
    if mode == "default": 
        train_slices, train_slices_masks, _, val_slices, val_slices_masks, _, test_slices, test_slices_masks, _ = getData()
            
        train, test, val, trainMasks, testMasks, valMasks=prepare_CNN(train_slices, train_slices_masks, val_slices, val_slices_masks, test_slices, test_slices_masks)
        results, accuracy, dice, jaccard, preds_test_nodules=train_model(train, test, val, trainMasks, testMasks, valMasks)
        plot_loss(results)
        print("Test set: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice, jaccard, accuracy))
        
    
    if mode == "cross_val": 
        train_slices, train_slices_masks, _, val_slices, val_slices_masks, _, test_slices, test_slices_masks, _ = getData(mode="cross_val")
        
        for train_x, train_masks, val_x, val_masks in zip(train_slices, train_slices_masks, val_slices, val_slices_masks):
            
            train, test, val, trainMasks, testMasks, valMasks=prepare_CNN(train_x, train_masks, val_x, val_masks, test_slices, test_slices_masks)
            results, accuracy, dice, jaccard, preds_test_nodules=train_model(train, test, val, trainMasks, testMasks, valMasks)
            plot_loss(results)
            print("Test set: The dice value is %.2f and the jaccard value is %.2f. The accuracy is %.2f" % (dice, jaccard, accuracy))

    

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
    
    val_slices_masks = val_slices_masks.reshape(-1, 64,64, 1)
    
   
    

    return train_slices, test_slices, val_slices, train_slices_masks, test_slices_masks, val_slices_masks

#%%
"""
conv2d_block: Definition of the convolution layer for the model

Arguments:  input_tensor-  input image
            n_filters - number of filters
            kernel_size - speaks for it self kernels of convolution
            batch norm - batch normalization - True when it does it
            
return: model after convolutions and relu activation
"""

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

"""
get_unet: performs U-net model on image

Arguments:  input_tensor-  input image
            n_filters - number of filters
            kernel_size - speaks for it self kernels of convolution
            batch norm - batch normalization - True when it does it
            
return: trained model
"""

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

def train_model(train_slices, test_slices, val_slices, train_slices_masks, test_slices_masks, val_slices_masks):
    # define parameters
    im_width = 64
    im_height = 64
    epochs=130
    batch=len(train_slices)
    
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=3, dropout=0.05, batchnorm=True)
    
    
    # define data preparation- keras data augmentation
    image_gen = ImageDataGenerator(rotation_range=180,width_shift_range=.15,height_shift_range=.15,horizontal_flip=True, vertical_flip=True)
    # fit parameters from data
    image_gen.fit(train_slices, augment=True)
    
    
    model.compile(optimizer=Adam(), loss=IoU_loss)
    #model.summary()
    
    callbacks = [
        ModelCheckpoint('model2dsegmentação.h5', verbose=0, save_best_only=True, save_weights_only=True)
    ]
    
    results = model.fit_generator(image_gen.flow(train_slices, train_slices_masks, batch_size=batch),steps_per_epoch=10, epochs=epochs, callbacks=callbacks, verbose=0, validation_data=(val_slices, val_slices_masks))
    
    model.load_weights('model2dsegmentação.h5')
    
    
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
    preds_test = model.predict(test_slices, verbose=0)
    preds_test_nodules = (preds_test >best_treshold).astype(np.uint8)
    
    preds_test_nodules=preds_test_nodules.reshape(-1,64,64)
    #test_slices_masks=test_slices_masks.reshape(-1,51,51)
    
    #cut the border previously used to match the ground truth
    border_size_top_right=6
    border_size_bottom_left=6
    preds_test_nodules=[nodule[border_size_top_right:-(border_size_top_right+1),border_size_bottom_left:-(border_size_bottom_left+1)] for nodule in preds_test_nodules]
    
    
    #Aplly morphologic operation to close some holes on predicted images
    preds_test_nodules=closing(preds_test_nodules)
    
    accuracy, dice, jaccard = confusionMatrix(np.hstack(np.hstack(preds_test_nodules)), np.hstack(np.hstack(test_slices_masks)))

    return results, accuracy, dice, jaccard, preds_test_nodules


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
"""
confusionMatrix - calculates the confusion matriz given a prediction and a labeled array
=====================
Arguments: predictions: array with predicted results
            labels: corresponding ground true
Return: confusion matrix
"""
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
    
    plt.figure()
    plt.title("Training and Validation loss")
    plt.plot(results.history["loss"], 'bo', label="loss")
    plt.plot(results.history["val_loss"],'b', label="val_loss")
    plt.legend();


#%%
run_segmentation_CNN(mode = "cross_val")
