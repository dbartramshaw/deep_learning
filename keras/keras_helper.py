#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Keras Helper Notes
"""

import keras
import numpy as np
import pandas as pd

from keras.models import Sequential,load_model, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers.core import Permute, Reshape
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import Regularizer
from keras import regularizers
from keras.utils import to_categorical
from keras.optimizers import Adam, Nadam, Adadelta



###################################
# Add vs Assign
###################################
dimentions = 100
window_size = 16
conv_feat_map = 8
model = Sequential()


#----------------------------
# Batch Normalisation  (Add)
model.add(BatchNormalization(input_shape=(dimentions,window_size,1)))

# Batch Normalisation  (Assign)
input_shape = Input(shape=(dimentions,window_size,1), name='input_name')
x = BatchNormalization()(input_shape)


#----------------------------
# Conv2D (Add)
model.add(Conv2D(conv_feat_map, kernel_size=(1, 5),activation='relu',padding='same'))

# Conv2D (Assign)
x = Conv2D(conv_feat_map, kernel_size=(1, 5),activation='relu',padding='same')(x)


#----------------------------
# Dropout (Add)
model.add(Dropout(0.5))
# Dropout (Assign)
x = Dropout(0.5)(x)


#----------------------------
# Swapping Dimentions (Add)
model.add(Permute((2, 1, 3)))

# Swapping Dimentions (Assign)
x = Permute((2, 1, 3))(x)




###################################
# Checkpointer - save best model
###################################
checkpointer = ModelCheckpoint(filepath="Efusion_fp_aud_dominance.hdf5"
                             , monitor='val_mean_squared_error' #Change to required metric
                             , verbose=1
                             , save_best_only=True)

y_train.shape
H = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=(x_test, y_test)
            ,callbacks=[checkpointer]



###################################
# Linear Predicitons
###################################
output = Dense(1, activation='linear')(z)
model = Model(inputs=input_1, outputs=output)
model.compile(loss=keras.losses.mean_squared_error,
              optimizer='adam',
              metrics=['mse'])



###################################
# Multiple Inputs
###################################
input_1 = Input(shape=(feat_dim,WINDOW_SIZE,1), name='X_train_images')
x = BatchNormalization()(input_1)
x = ...

input_2 = Input(shape=(timesteps, data_dim), name='X_train_audio')
x = BatchNormalization()(input_2)
x = ...

model = Model(inputs=[input_1, input_2], outputs=output)

H = model.fit([x_train_images, x_train_audio], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=([x_train_images, x_test_audio], y_test)
            ,callbacks=[checkpointer]
            )
