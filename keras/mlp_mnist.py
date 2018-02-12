#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
####################################
Keras Intro - MNIST
####################################

#http://parneetk.github.io/blog/neural-networks-in-keras/
In this notebook, we will learn to:

 - import MNIST dataset and visualize some example images
 - define deep neural network model with single as well as multiple hidden layers
 - train the model and plot the accuracy or loss at each epoch
 - study the effect of varying the learning rate, batch size and number of epochs
 - use SGD and Adam optimizers
 - save model weights every 10 epochs
 - resume training by loading a saved model
 - earlystop the training if there is negligiable improvement in the performance
"""

# Use GPU for Theano, comment to use CPU instead of GPU
# Tensorflow uses GPU by default
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu, floatX=float32"

# If using tensorflow, set image dimensions order
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# PACKAGE SETUP
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
% matplotlib inline
np.random.seed(2017)

## LOAD DATA
from keras.datasets import mnist
(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
_, img_rows, img_cols =  train_features.shape
num_classes = len(np.unique(train_labels))
num_input_nodes = img_rows*img_cols
print ("Number of training samples: %d"%train_features.shape[0])
print ("Number of test samples: %d"%test_features.shape[0])
print ("Image rows: %d"%train_features.shape[1])
print ("Image columns: %d"%train_features.shape[2])
print ("Number of classes: %d"%num_classes)



## SHOW SOME EXAMPLES
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    features_idx = train_features[train_labels[:]==i,:]
    ax.set_title("Num: " + str(i))
    plt.imshow(features_idx[1], cmap="gray")
plt.show()

## PREPROCESSING
# reshape images to column vectors
train_features.shape # before (60000, 28, 28)
train_features = train_features.reshape(train_features.shape[0], img_rows*img_cols)
train_features.shape #after (60000, 784)
test_features = test_features.reshape(test_features.shape[0], img_rows*img_cols)

# convert class labels to binary class labels (One hot encoding)
train_labels[0:10]
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)
test_labels[0:10]



# MODEL STRUCTURE
# Define a Neural Network Model with a Single Hidden Layer
def simple_nn():
    # initialize model
    model = Sequential()
    # add an input layer and a hidden layer
    model.add(Dense(100, input_dim = num_input_nodes))
    # add activation layer to add non-linearity
    model.add(Activation('sigmoid'))
    # to add ReLu instead of sigmoid: model.add(Activation('relu'))
    # combine above 2 layers: model.add(Dense(100, input_dim=784),Activation('sigmoid'))
    # add output layer
    model.add(Dense(num_classes))
    # add softmax layer
    model.add(Activation('softmax'))
    return model


# DEFINE MODEL
model = simple_nn()
# define optimizer
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
# print model information
model.summary()


#TRAIN THE MODEL
start = time.time()
model_info = model.fit(train_features, train_labels,
                        batch_size=64,
                        epochs=10,
                        #verbose=2,
                        validation_split=0.2)
                        #shuffle=True)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))

# stored history
# model_info.history['acc']

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

'''
Observation:
Both training and validation accuracy increase as the number of epochs increase. More information is learned in each epoch.
'''
plot_model_history(model_info)



# TEST THE MODEL
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


print("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))

# result = model.predict(test_features)
# predicted_class = np.argmax(result, axis=1)
# true_class = np.argmax(test_labels, axis=1)
# num_correct = np.sum(predicted_class == true_class)
# accuracy = float(num_correct)/result.shape[0]
# (accuracy * 100)



'''
#############################
# Vary the learning rate
#############################
Observation:
If the learning rate is decreased, less information is learned in each epoch and more epochs are required to learn a good model.
If the learning rate is increased, more information is learned in each epoch and less epochs are required to learn a good model.
When using SGD, learning rate needs to be decided emperically for a given dataset.
'''
# decrease the learning rate
# define model
model = simple_nn()
# define optimizer
sgd = SGD(lr=0.001) #sgd = SGD(lr=0.05)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
# train the model
start = time.time()
model_info = model.fit(train_features, train_labels,
                        batch_size=64,
                        epochs=10,
                        verbose=0,
                        validation_split=0.2)
end = time.time()
# plot model history
plot_model_history(model_info)
print("Model took %0.2f seconds to train"%(end - start))
# compute test accuracy
print("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))



'''
#############################
# Use the Adam optimiser
#############################
Observation:
Using Adam optimizer, we don’t need to specify a learning rate.
However, the training time increases. Refer this tutorial for an interesting comparison of optimizers.

'''
# Define model
model = simple_nn()
# define optimizer, loss function
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# Train the model
start = time.time()
model_info = model.fit(train_features, train_labels,
                       batch_size=64,
                       epochs=10,
                       verbose=0,
                       validation_split=0.2)
end = time.time()
# plot model history
plot_model_history(model_info)
print("Model took %0.2f seconds to train"%(end - start))
# compute test accuracy
print("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))




'''
#############################
# Vary the bacth size
#############################
Observation:
Increasing the batch size decreases the training time but reduces the rate of learning.
Batches are the number of observations learning the weights at once
'''
# increase the batch size
# define model
model = simple_nn()
# define optimizer
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# train the model
start = time.time()
model_info = model.fit(train_features, train_labels, batch_size=128, \
                       nb_epoch=10, verbose=0, validation_split=0.2)
end = time.time()
# plot model history
plot_model_history(model_info)
print("Model took %0.2f seconds to train"%(end - start))
# compute test accuracy
print("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))



'''
#############################
# Deep NN
#############################
'''
def deep_nn():
    # Define a deep neural network
    model = Sequential()
    model.add(Dense(512, input_dim=num_input_nodes))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model






###
