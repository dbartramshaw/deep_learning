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
