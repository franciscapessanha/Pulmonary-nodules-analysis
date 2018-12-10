# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:34:46 2018

@author: Hugo Barros
"""
from get_data import getData
import numpy as np
import matplotlib.pyplot as plt

#%%
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
import keras
#%% Import Data
train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()

def prepare_CNN():
    train_slices, train_slices_masks, y_train, test_slices, test_slices_masks, y_test , val_slices, val_slices_masks, y_val = getData()

    #we want the information aout the nodules therefore let's assume segmentation is done
    train_slices = [np.multiply(train_slices_masks[i],train_slices[i]) for i in range(len(train_slices))]
    test_slices = [np.multiply(test_slices_masks[j],test_slices[j]) for j in range(len(test_slices))]
    val_slices = [np.multiply(val_slices_masks[k],val_slices[k]) for  k in range(len(val_slices))]

    
    train_slices = np.asarray(train_slices)
    test_slices = np.asarray(test_slices)
    val_slices = np.asarray(val_slices)
    
    
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


#%%
train_slices, test_slices, val_slices, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot=prepare_CNN()


#def CNN(train_slices, test_slices, val_slices, train_Y_one_hot, test_Y_one_hot, val_Y_one_hot)
# define data preparation
image_gen = ImageDataGenerator(rotation_range=90,width_shift_range=.15,height_shift_range=.15,horizontal_flip=True)
# fit parameters from data
image_gen.fit(train_slices, augment=True)

#%%===========================================
#Defenition of parameters
batch_size = 92
epochs = 10
num_classes = 3
class_weights = {0: 2.05, 1: 1.875, 2: 1.0} # Because of unbaleced training set - classes with less images have more weight
#during the training, ReLU units can "die". This can happen when a large gradient flows through a ReLU neuron: it can cause the weights to update in such a way that the neuron will never activate on any data point again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. Leaky ReLUs attempt to solve this: the function will not be zero but will instead have a small negative slope.
fashion_model = Sequential()
fashion_model.add(Conv2D(16, kernel_size=(5, 5),activation='relu', input_shape=(51,51,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((3, 3),padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(Dropout(0.2))


fashion_model.add(Conv2D(64, (5, 5),activation='relu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((3, 3),padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(Dropout(0.2))


fashion_model.add(Flatten())
fashion_model.add(Dense(1152, activation='relu'))         
fashion_model.add(Dropout(0.2))
fashion_model.add(Dense(256, activation='relu'))         
fashion_model.add(Dropout(0.2))
fashion_model.add(Dense(num_classes, activation='softmax'))


fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
fashion_train = fashion_model.fit_generator(image_gen.flow(train_slices, train_Y_one_hot, batch_size=batch_size),steps_per_epoch=10, epochs=epochs,verbose=1,validation_data=(val_slices, val_Y_one_hot), class_weight=class_weights)

#fashion_train = fashion_model.fit(train_slices, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_slices, val_Y_one_hot))
#fashion_model.summary()
test_eval = fashion_model.evaluate(test_slices, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = fashion_model.predict(test_slices)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

    
    #return

#%% Show

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']

loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
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





