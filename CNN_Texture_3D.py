# -*- coding: utf-8 -*-

from get_data import getData
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
   
#%%
def run_CNN_segmentation_3D():
   
    train_volumes, train_volumes_masks, y_train, val_volumes, val_volumes_masks, y_val, test_volumes, test_volumes_masks, y_test = getData(mode = "default", type_ = "volume")
    train_volumes, test_volumes, val_volumes, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot=prepare_CNN(train_volumes, train_volumes_masks, y_train, test_volumes, test_volumes_masks, y_test , val_volumes, val_volumes_masks, y_val)
    predicted_classes, fashion_train=train_model(train_volumes, test_volumes, val_volumes, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot) # model predicted for test set
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
    * train_volumes, test_volumes, val_volumes - images of train, test and validation sets after normalization and reshaped for input on the CNN
    train_Y_one_hot, test_Y_one_hot, val_Y_one_hot - classes in the format of one-hot-vector (1,0,0)
"""
def prepare_CNN(train_volumes, train_volumes_masks, y_train, test_volumes, test_volumes_masks, y_test , val_volumes, val_volumes_masks, y_val):

    #we want the information aout the nodules therefore let's assume segmentation is done
    train_volumes = [np.multiply(train_volumes_masks[i],train_volumes[i]) for i in range(len(train_volumes))]
    test_volumes = [np.multiply(test_volumes_masks[j],test_volumes[j]) for j in range(len(test_volumes))]
    val_volumes = [np.multiply(val_volumes_masks[k],val_volumes[k]) for  k in range(len(val_volumes))]
    
    mean_int=np.mean(train_volumes)
    std_int=np.std(train_volumes)
    
    train_volumes = (train_volumes - mean_int)/std_int
    val_volumes = (val_volumes - mean_int)/std_int
    test_volumes = (test_volumes - mean_int)/std_int
    
    train_volumes = np.asarray(train_volumes)
    test_volumes = np.asarray(test_volumes)
    val_volumes = np.asarray(val_volumes)
    
    
    train_volumes = train_volumes.reshape(-1, 51,51, 51, 1)
    test_volumes = test_volumes.reshape(-1, 51,51, 51, 1)
    val_volumes = val_volumes.reshape(-1, 51,51, 51, 1)
    
    train_volumes = train_volumes.astype('float32')
    test_volumes = test_volumes.astype('float32')
    val_volumes = val_volumes.astype('float32')
    
    # Change the labels from categorical to one-hot encoding necessarty to CNN      
    train_Y_one_hot = to_categorical(y_train)
    test_Y_one_hot = to_categorical(y_test)
    val_Y_one_hot = to_categorical(y_val)
    
    return  train_volumes, test_volumes, val_volumes, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot

#%%
"""
train_model
===============
train a CNN model and fits the train set on it. It tests the validation set and predicts for test set

Arguments: train_volumes, test_volumes, val_volumes train_Y_one_hot, test_Y_one_hot, val_Y_one_hot - images and classes of all 3 sets
        
Returns:
    * predicted_classes - vector with the classes predicted for the test set
    * fashion_train - model trained
"""
def train_model(train_volumes, test_volumes, val_volumes, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot):
    #Defenition of parameters
    batch_size = 1
    epochs = 40
    
    num_classes = 3
    class_weights = {0: 2.05, 1: 1.875, 2: 1.0} # Because of unbaleced training set - classes with less images have more weight
    
    
    #during the training, ReLU units can "die". This can happen when a large gradient flows through a ReLU neuron: 
    #it can cause the weights to update in such a way that the neuron will never activate on any data point again. 
    #If this happens, then the gradient flowing through the unit will forever be zero from that point on. 
    #Leaky ReLUs attempt to solve this: the function will not be zero but will instead have a small negative slope.
    fashion_model = Sequential()
    fashion_model.add(Conv3D(16, kernel_size=(3, 3,3),activation='relu', input_shape=(51,51,51,1),padding='same'))
    fashion_model.add(MaxPooling3D((2, 2,2),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))
    
    
    fashion_model.add(Conv3D(64, (3, 3,3),activation='relu',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling3D((2, 2,2),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))
    
    fashion_model.add(Conv3D(64, (3, 3,3),activation='relu',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling3D((2, 2,2),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))
    
    fashion_model.add(Flatten())
    fashion_model.add(Dense(1152, activation='relu'))         
    fashion_model.add(Dropout(0.2))
    fashion_model.add(Dense(256, activation='relu'))         
    fashion_model.add(Dropout(0.2))
    fashion_model.add(Dense(num_classes, activation='softmax'))
    
    
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    #fashion_model.summary()
    
    callbacks = [
        EarlyStopping(patience=10, verbose=0),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=0),
        ModelCheckpoint('model3dtexture.h5', verbose=0, save_best_only=True, save_weights_only=True)
    ]
    
    
    fashion_train=fashion_model.fit(train_volumes, train_Y_one_hot, batch_size=batch_size,epochs=epochs, callback=callbacks,verbose=1, validation_data=(val_volumes, val_Y_one_hot), class_weight = class_weights)
    #fashion_model.summary()
    
    fashion_model.load_weights('model3dtextura.h5')
    
    predicted_classes = fashion_model.predict(test_volumes)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

    return predicted_classes, fashion_train


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
    accuracy = fashion_train.history['acc']
    val_accuracy = fashion_train.history['val_acc']
    epochs = range(len(accuracy))
    loss = fashion_train.history['loss']
    val_loss = fashion_train.history['val_loss']
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    
#%%
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

def separateClasses(predictSVM):
    solid =[] # label 2
    sub_solid = [] # label 1
    non_solid = [] # label 0
    for j in range(len(predictSVM)):
        if predictSVM[j] == 0:
            non_solid.append(1)
        else: 
            non_solid.append(0)
            
        if predictSVM[j] == 1:
            sub_solid.append(1)
        else: 
            sub_solid.append(0)
            
        if predictSVM[j] == 2:
            solid.append(1)
        else: 
            solid.append(0)
            
    return solid, sub_solid, non_solid

#%%
run_CNN_segmentation_3D()