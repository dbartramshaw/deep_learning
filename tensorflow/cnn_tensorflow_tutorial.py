#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" ###################################################
     Convolutional Neural Networks in Tensorflow
"'" ###################################################
    # https://www.tensorflow.org/tutorials/layers

MNIST dataset CNN architecture:

 1) Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions),
    with ReLU activation function

 2) Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2
    (which specifies that pooled regions do not overlap)

 3) Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function

 4) Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2

 5) Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4
    (probability of 0.4 that any given element will be dropped during training)

 6) Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).



The tf.layers module contains methods to create each of the three layer types above:

- conv2d().
Constructs a two-dimensional convolutional layer. Takes number of
filters, filter kernel size, padding, and activation function as arguments.

- max_pooling2d().
Constructs a two-dimensional pooling layer using the
max-pooling algorithm. Takes pooling filter size and stride as arguments.

- dense().
Constructs a dense layer. Takes number of neurons and activation function as arguments.

"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer = [batch_size,image_width,image_height,channels]
  # batch_size of -1 dynamic based on the number of input values in features["x"]
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  ########################
  # Model Framework      #
  ########################

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      # How many kernals
      filters=32,
      # the size of the learned features
      kernel_size=[5, 5],
      # same ensures each conv is same size. Will pad right if odd number
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # stride of 2 indicates the subregions extracted by the filter should be separated by 2
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  # Before we connect the layer, however, we'll flatten our feature map (pool2) to shape [batch_size, features]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  # we choose 1,024 neurons (arbitrary)
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  # 40% of the elements will be randomly dropped out during training.
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  # returns raw values for our predictions.
  logits = tf.layers.dense(inputs=dropout, units=10)
  # Our final output tensor of the CNN, logits, has shape [batch_size, 10]

  ########################
  # Generate Predictions #
  ########################
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  ########################
  # Calculate Loss       #
  ########################

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



if __name__ == "__main__":
  tf.app.run()
