#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
########################
Auto Encoders
########################

Auto-Encoder Example
Build a 2 layers auto-encoder with TensorFlow to compress images to a lower latent space and then reconstruct them.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



"""
########################
Data import
########################
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Data from tensorflow
training_data = mnist.train.images
training_labels = mnist.train.labels
training_data = mnist.test.images
training_labels = mnist.test.labels
validation_data = mnist.validation.images
validation_labels = mnist.validation.labels


# Reshaping is need be
# training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
# training_results = [vectorized_result(y) for y in tr_d[1]]
# training_data = zip(training_inputs, training_results)



def show_img_vec(image_vec,label_vec):

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(image_vec)
    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('Label is {label}'.format(label=label_vec))
    plt.imshow(pixels, cmap='gray')
    plt.show()

# single example
image_vec = mnist.train.images[0]
label_vec = mnist.train.labels[0]
show_img_vec(image_vec,label_vec)




"""
########################
Data import
########################
"""
# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])


weights = {
                'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
                'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
                'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
                'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
            }
dir(weights['encoder_h1'])

biases = {
                'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
                'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
                'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
                'decoder_b2': tf.Variable(tf.random_normal([num_input])),
            }




"""
########################
Model Setup
########################
"""
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add
                                    (
                                    tf.matmul(x, weights['encoder_h1'])
                                    ,biases['encoder_b1']
                                    )
                            )
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add
                                    (
                                    tf.matmul(layer_1, weights['encoder_h2'])
                                    ,biases['encoder_b2']
                                    )
                            )
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()



"""
########################
Train Model
########################
"""
# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)

# Training
for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = mnist.train.next_batch(batch_size)

    # Run optimization op (backprop) and cost op (to get loss value)
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    # Display logs per step
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))


# Testing
# Encode and decode images from test set and visualize their reconstruction.
n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i in range(n):
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    # Encode and decode the digit image
    g = sess.run(decoder_op, feed_dict={X: batch_x})

    # Display original images
    for j in range(n):
        # Draw the generated digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
    # Display reconstructed images
    for j in range(n):
        # Draw the generated digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

## Task!
## Can we visualise the layers????





"""

########################################################################
AUTOENCODERS IN KERAS
########################################################################

We'll start simple, with a single fully-connected neural layer as encoder and as decoder:
"""
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)


#######################################################
# create encoder model:
#######################################################
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)


# As well as the decoder model:
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


#########################
# Train
#########################
# Now let's train our autoencoder to reconstruct MNIST digits.
# First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Let's prepare our input data. We're using MNIST digits, and we're discarding the labels (since we're only interested in encoding/decoding the input images).
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

# We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train
print x_test.shape


# Now let's train our autoencoder for 50 epochs:
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


favorite_color = pickle.load( open( "save.p", "rb" ) )
