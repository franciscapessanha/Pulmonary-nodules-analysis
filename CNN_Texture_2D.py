# -*- coding: utf-8 -*-

from get_data import getData
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras

#%%
"""
Runs all code regarding texture 2D for CNN aproach

arguments: mode - default runs 1 time or cross_val runs 5 times with different sets of trains/validation

returns: void
"""
def run_CNN_segmentation_2D(mode = "default"):
    if mode == "default": 
        train_slices, train_slices_masks, y_train, val_slices, val_slices_masks, y_val, test_slices, test_slices_masks, y_test = getData(mode = "default")
        train_slices, test_slices, val_slices, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot=prepare_CNN(train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val)
        predicted_classes, predicted_classes_val, fashion_train=train_model(train_slices, test_slices, val_slices, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot) # model predicted for test set
        show_loss(fashion_train)
       
        solid_pred, sub_solid_pred, non_solid_pred = separateClasses(predicted_classes)
        solid_label, sub_solid_label, non_solid_label = separateClasses(y_test)
        
        accuracy_solid, precision_solid, recall_solid, auc_solid = getPerformanceMetrics(solid_pred, solid_label)
        accuracy_sub, precision_sub, recall_sub, auc_sub = getPerformanceMetrics(sub_solid_pred, sub_solid_label)
        accuracy_non, precision_non, recall_non, auc_non = getPerformanceMetrics(non_solid_pred, non_solid_label)
        print("Solid texture: The accuracy value is %.2f and the precision value is %.2f. The recall is %.2f  and the auc is %.2f" % (accuracy_solid, precision_solid, recall_solid, auc_solid))
        print("Sub solid texture: The accuracy value is %.2f and the precision value is %.2f. The recall is %.2f  and the auc is %.2f" % (accuracy_sub, precision_sub, recall_sub, auc_sub))
        print("Non solid texture: The accuracy value is %.2f and the precision value is %.2f. The recall is %.2f  and the auc is %.2f" % (accuracy_non, precision_non, recall_non, auc_non))
   
        
    
    elif mode == "cross_val":
        train_slices, train_slices_masks, y_train , val_slices, val_slices_masks, y_val, test_slices, test_slices_masks, y_test = getData(mode = "cross_val")
        
        for train_x, train_masks, train_y, val_x, val_masks, val_y  in zip(train_slices, train_slices_masks, y_train , val_slices, val_slices_masks, y_val):
            
            train, test, val, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot=prepare_CNN(train_x, train_masks, train_y, test_slices, test_slices_masks, y_test, val_x, val_masks, val_y)
            predicted_classes, predicted_classes_val, fashion_train=train_model(train, test, val, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot)
        
            show_loss(fashion_train)
            solid_pred, sub_solid_pred, non_solid_pred = separateClasses(predicted_classes)
            solid_label, sub_solid_label, non_solid_label = separateClasses(y_test)
              
            accuracy_solid, precision_solid, recall_solid, auc_solid = getPerformanceMetrics(solid_pred, solid_label)
            accuracy_sub, precision_sub, recall_sub, auc_sub = getPerformanceMetrics(sub_solid_pred, sub_solid_label)
            accuracy_non, precision_non, recall_non, auc_non = getPerformanceMetrics(non_solid_pred, non_solid_label)
            print("Solid texture: The accuracy value is %.2f and the precision value is %.2f. The recall is %.2f  and the auc is %.2f" % (accuracy_solid, precision_solid, recall_solid, auc_solid))
            print("Sub solid texture: The accuracy value is %.2f and the precision value is %.2f. The recall is %.2f  and the auc is %.2f" % (accuracy_sub, precision_sub, recall_sub, auc_sub))
            print("Non solid texture: The accuracy value is %.2f and the precision value is %.2f. The recall is %.2f  and the auc is %.2f" % (accuracy_non, precision_non, recall_non, auc_non))
   
#%%
"""
prepare CNN
===============
prepares the data for input in the CNN

Arguments:
    
Returns:
    * train_slices, test_slices, val_slices - images of train, test and validation sets after normalization and reshaped for input on the CNN
    train_Y_one_hot, test_Y_one_hot, val_Y_one_hot - classes in the format of one-hot-vector (1,0,0)
"""
def prepare_CNN(train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val):
    

    #we want the information aout the nodules therefore let's assume segmentation is done
    train_slices = [np.multiply(train_slices_masks[i],train_slices[i]) for i in range(len(train_slices))]
    test_slices = [np.multiply(test_slices_masks[j],test_slices[j]) for j in range(len(test_slices))]
    val_slices = [np.multiply(val_slices_masks[k],val_slices[k]) for  k in range(len(val_slices))]
   
    mean_int=np.mean(train_slices)
    std_int=np.std(train_slices)
    
    train_slices = (train_slices - mean_int)/std_int
    val_slices = (val_slices - mean_int)/std_int
    test_slices = (test_slices - mean_int)/std_int
    
    
    train_slices = train_slices.reshape(-1, 51,51, 1)
    test_slices = test_slices.reshape(-1, 51,51, 1)
    val_slices = val_slices.reshape(-1, 51,51, 1)
    
    train_slices = train_slices.astype('float32')
    test_slices = test_slices.astype('float32')
    val_slices = val_slices.astype('float32')
    
    # Change the labels from categorical to one-hot encoding necessarty to CNN      
    train_Y_one_hot = to_categorical(y_train)
    test_Y_one_hot = to_categorical(y_test)
    val_Y_one_hot = to_categorical(y_val)

    return train_slices, test_slices, val_slices, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot

#%%===========================================
    """
train_model
===============
train a CNN model and fits the train set on it. It tests the validation set and predicts for test set

Arguments: train_slices, test_slices, val_slices, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot - images and classes of all 3 sets
        
Returns:
    * predicted_classes - vector with the classes predicted for the test set
    * fashion_train - model trained
"""
def train_model(train_slices, test_slices, val_slices, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot):
    #Defenition of parameters
    batch_size = 92
    epochs = 40
    num_classes = 3
    class_weights = {0: 2.05, 1: 1.875, 2: 1.0} # Because of unbaleced training set - classes with less images have more weight the number of images of class 2is arround 2times the others
    
    # define data preparation- keras data augmentation
    image_gen = ImageDataGenerator(rotation_range=180,width_shift_range=.15,height_shift_range=.15,horizontal_flip=True, vertical_flip=True)
    # fit parameters from data
    image_gen.fit(train_slices, augment=True)
    
    #during the training, ReLU units can "die". This can happen when a large gradient flows through a ReLU neuron: it can cause the weights to update in such a way that the neuron will never activate on any data point again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. Leaky ReLUs attempt to solve this: the function will not be zero but will instead have a small negative slope.
    fashion_model = Sequential()
    fashion_model.add(Conv2D(16, kernel_size=(5, 5), input_shape=(51,51,1),padding='same', activation='relu'))
    fashion_model.add(MaxPooling2D((3, 3), padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))
    
    
    fashion_model.add(Conv2D(64, (5, 5), padding='same',activation='relu'))
    fashion_model.add(MaxPooling2D((3, 3),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.4))
    
    fashion_model.add(Flatten())
    fashion_model.add(Dense(1152, activation='relu'))         
    fashion_model.add(Dropout(0.5))
    fashion_model.add(Dense(256, activation='relu'))         
    fashion_model.add(Dropout(0.5))
    fashion_model.add(Dense(num_classes, activation='softmax'))
    
    callbacks = [
        ModelCheckpoint('model2dtextura.h5', verbose=0, save_best_only=True, save_weights_only=True)
    ]
    
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam())
    #fashion_model.summary()
    fashion_train = fashion_model.fit_generator(image_gen.flow(train_slices, train_Y_one_hot, batch_size=batch_size),steps_per_epoch=10, epochs=epochs,callbacks=callbacks, verbose=0,validation_data=(val_slices, val_Y_one_hot), class_weight=class_weights)

    fashion_model.load_weights('model2dtextura.h5')
    
    predicted_classes = fashion_model.predict(test_slices)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    
    predicted_classes_val = fashion_model.predict(val_slices)
    predicted_classes_val = np.argmax(np.round(predicted_classes_val),axis=1)
    
    return predicted_classes, predicted_classes_val, fashion_train



#%%
"""
show_loss_accuracy
===============
shows the cross entropy loss and the accuracy of train and validation sets during each epoch

Arguments: fashion_train - model trained
    
Returns:
    *void
"""
def show_loss(fashion_train):
    # Show
    loss = fashion_train.history['loss']
    val_loss = fashion_train.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
#%%
    """
Evaluation
===============================================================================
"""

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
getPerformanceMetrics- Calculates accuracy, precision, recall, auc for evaluation given an array of predictions and the corresponding ground true
=========================
Arguments: 
            predictions- array with predicted results
            labels-  corresponding ground true
Return: accuracy, precision, recall, auc - evaluation metrics
"""
def getPerformanceMetrics(predictions, labels):
    c_matrix = confusionMatrix(predictions, labels)
    
    true_positives = c_matrix[0,0]
    false_negatives = c_matrix[0,1]
    false_positives = c_matrix[1,0]
    true_negatives = c_matrix[1,1]

    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    precision = (true_positives)/(true_positives + false_positives + 10**-12)
    
    recall = (true_positives)/(true_positives + false_negatives)
    #matrix = np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    fp_rate, tp_rate, thresholds = metrics.roc_curve(labels, predictions, pos_label = 1)
    auc = metrics.auc(fp_rate, tp_rate)
    
    return accuracy, precision, recall, auc

"""
separateClasses - separates classes in 3 vector, one for each classes, 
                    where the values=1 correspondes to the predicted true results and 0 false predicted result
============================                    
Arguments: 
        predict- array with multiple classes 
Returns: solid, sub_solid, non_solid - binary vectors of each class
"""

def separateClasses(predict):
    solid =[] # label 2
    sub_solid = [] # label 1
    non_solid = [] # label 0
    for j in range(len(predict)):
        if predict[j] == 0:
            non_solid.append(1)
        else: 
            non_solid.append(0)
            
        if predict[j] == 1:
            sub_solid.append(1)
        else: 
            sub_solid.append(0)
            
        if predict[j] == 2:
            solid.append(1)
        else: 
            solid.append(0)
            
    return solid, sub_solid, non_solid

#%%

run_CNN_segmentation_2D(mode = "cross_val")


