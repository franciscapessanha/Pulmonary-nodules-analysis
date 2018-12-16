# -*- coding: utf-8 -*-

from get_data import getData
import numpy as np


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
   
#%%


train_volumes, train_masks, val_volumes, val_masks, test_volumes, test_masks = getData(mode = "default", type = "volume")

#we want the information aout the nodules therefore let's assume segmentation is done
train_volumes = [np.multiply(train_masks[i],train_volumes[i]) for i in range(len(train_volumes))]
test_volumes = [np.multiply(test_masks[j],test_volumes[j]) for j in range(len(test_volumes))]
val_volumes = [np.multiply(val_masks[k],val_volumes[k]) for  k in range(len(val_volumes))]

#%%
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

#%%
import keras
#Defenition of parameters
batch_size = 1
epochs = 10
num_classes = 3
class_weights = {0: 2.05, 1: 1.875, 2: 1.0} # Because of unbaleced training set - classes with less images have more weight
#during the training, ReLU units can "die". This can happen when a large gradient flows through a ReLU neuron: it can cause the weights to update in such a way that the neuron will never activate on any data point again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. Leaky ReLUs attempt to solve this: the function will not be zero but will instead have a small negative slope.
fashion_model = Sequential()
fashion_model.add(Conv3D(16, kernel_size=(3, 3,3), input_shape=(51,51,51,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling3D((2, 2,2),padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(Dropout(0.2))


fashion_model.add(Conv3D(64, (3, 3,3),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling3D((2, 2,2),padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv3D(64, (3, 3,3),padding='same'))
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
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
fashion_model.fit(train_volumes, train_Y_one_hot, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1, validation_data=(val_volumes, val_Y_one_hot), class_weight = {0: 2.05, 1: 1.875, 2: 1.0})
#fashion_model.summary()
test_eval = fashion_model.evaluate(test_volumes, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = fashion_model.predict(test_volumes)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
