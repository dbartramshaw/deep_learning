#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Tensorflow intro
"""

import tensorflow as tf
import numpy as np
#http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

tf.Graph().as_default()
'''
------------------------------------
# 1.1 Creating constants and variables
------------------------------------
The first element in both is the value to be assigned the constant
variable when it is initialised.
The second is an optional name string which can be used to label the constant
variable – this is handy TensorFlow will infer the type of the constant
variable from the initialised value, but it can also be set explicitly using
the optional dtype argument.

It’s important to note that, as the Python code runs through these commands,
the variables haven’t actually been declared as they would have been if you
just had a standard Python declaration (i.e. b = 2.0).
Instead, all the constants, variables, operations and the computational graph
are only created when the initialisation commands are run.

THEY'RE NOT ACTUALLY CREATED UNTIL INTIALISATION RUNS
'''

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# create TensorFlow variables
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, name='c')
d = tf.Variable(1.0, name='d', dtype=tf.float32) #choose type



'''
------------------------------------
# Creating TensorFlow operations:
------------------------------------
'''
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')



'''
------------------------------------
# Setup the variable initialisation
# Run Session
------------------------------------
Ok, so now we are all set to go.  To run the operations between the variables,
we need to start a TensorFlow session – tf.Session.  The TensorFlow session is
an object where all operations are run.  Using the with Python syntax, we can
run the graph with the following code:

with tf.Session() as sess:
    The first command within the with block is the initialisation,
    which is run with the, well, run command.

    Next we want to figure out what the variable a should be.
    All we have to do is run the operation which calculates a
    i.e. a = tf.multiply(d, e, name=’a’).

    !!Note that a is an operation, not a variable and therefore it can be run!!

    We do just that with the sess.run(a) command and assign the output to a_out,
    the value of which we then print out.

    Note something cool – we defined operations d and e which need to be
    calculated before we can figure out what a is.  However, we don’t have to
    explicitly run those operations, as TensorFlow knows what other operations
    and variables the operation a depends on, and therefore runs the necessary
    operations on its own.  It does this through its data flow graph which shows
    it all the required dependencies. Using the TensorBoard functionality,
    we can see the graph that TensorFlow created in this little program:
'''

# setup the variable initialisation
# - setup an object to initialise the variables and the graph structure:
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))
    d_out = sess.run(d)
    print("Variable d is {}".format(d_out))
    # You can view the graph using this - look for more detail
    #writer = tf.summary.FileWriter("output", sess.graph)


'''
------------------------------------
# 2.1 The TensorFlow placeholder
------------------------------------
Let’s also say that we didn’t know what the value of the array b would be during
the declaration phase of the TensorFlow problem (i.e. before the with
tf.Session() as sess) stage.  In this case, TensorFlow requires us to declare
the basic structure of the data by using the tf.placeholder variable declaration.
Let’s use it for b
'''

tf.reset_default_graph()
# create TensorFlow variables
const = tf.constant(2.0, name="const")

# create TensorFlow variables
# b = tf.Variable(2.0, name='b')
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')


init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))



'''
------------------------------------
# DBS Notes on recreating variables
------------------------------------
In the above example we had to run
                                        tf.reset_default_graph()
to reset all the variables because tensorflow was renaming b. (b_1)
This is because you cannot oevrwrite variables.

To get a round this you could create your own graph each time rather than using
the default graph. like so:

    g = tf.Graph()
    sess = tf.InteractiveSession(graph=g)
    with g.asdefault():
        # Put variable declaration and other tf operation
        # in the graph context
        ....
        b = tf.matmul(A, x)
        ....

     sess.run([b], ...)

found here: https://stackoverflow.com/questions/33765336/remove-nodes-from-graph-or-reset-entire-default-graph
'''





'''
--------------------------------------------------------------------------------
# 3.0 A Neural Network Example
--------------------------------------------------------------------------------
Now we’ll go through an example in TensorFlow of creating a simple three layer
neural network.

Notice the x input layer is 784 nodes corresponding to the 28 x 28 (=784) pixels,
and the y output layer is 10 nodes corresponding to the 10 possible digits.
Again, the size of x is (? x 784), where the ? stands for an as yet unspecified
number of samples to be input – this is the function of the placeholder variable.

'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.reset_default_graph()
# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])

# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

mnist
'''
------------------------------------
# Weight and Bias
------------------------------------
Now we need to setup the weight and bias variables for the three layer neural
network. There are always L-1 number of weights/bias tensors, where L is the
number of layers.  So in this case, we need to setup two tensors for each:

Ok, so let’s unpack the above code a little.  First, we declare some variables
for W1 and b1, the weights and bias for the connections between the input and
hidden layer.  This neural network will have 300 nodes in the hidden layer,
so the size of the weight tensor W1 is [784, 300].

We initialise the values of the weights using a random normal distribution with
a mean of zero and a standard deviation of 0.03.

TensorFlow has a replicated version of the numpy random normal function,
which allows you to create a matrix of a given size populated with random
samples drawn from a given distribution.  Likewise, we create W2 and b2
variables to connect the hidden layer to the output layer of the neural network.

'''
# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')


'''
------------------------------------
# calculate output of the hidden layer
------------------------------------
In the first line, we execute the standard matrix multiplication of the
weights (W1) by the input vector x and we add the bias b1.
The matrix multiplication is executed using the tf.matmul operation.
Next, we finalise the hidden_out operation by applying a rectified linear
unit activation function to the matrix multiplication plus bias.
Note that TensorFlow has a rectified linear unit activation already setup
for us, tf.nn.relu.
'''
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))


'''
------------------------------------
# Loss function
------------------------------------
We also have to include a cost or loss function for the optimisation
backpropagation to work on. Here we’ll use the cross entropy cost function,
represented by:J=−1m∑i=1m∑j=1ny(i)jlog(yj_(i))+(1–y(i)j)log(1–yj_(i))

We also clip to prevent log(0)
'''



y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))


'''
------------------------------------
# Loss function
------------------------------------
Here we are just using the gradient descent optimiser provided by TensorFlow.
We initialize it with a learning rate, then specify what we want it to do
i.e. minimise the cross entropy cost operation we created.
This function will then perform the gradient descent

TensorFlow has a library of popular neural network training optimisers,
see here. https://www.tensorflow.org/api_guides/python/train
'''
# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


'''
------------------------------------
# Variable Initialisation
------------------------------------
The correct prediction operation correct_prediction makes use of the TensorFlow
tf.equal function which returns True or False depending on whether to arguments
supplied to it are equal.

The tf.argmax function is the same as the numpy argmax function,
which returns the index of the maximum value in a vector / tensor.

Therefore, the correct_prediction operation returns a tensor of size (m x 1)
of True and False values designating whether the neural network has correctly
predicted the digit.

We then want to calculate the mean accuracy from this tensor
first we have to cast the type of the correct_prediction operation from a
Boolean to a TensorFlow float in order to perform the reduce_mean operation.
Once we’ve done that, we now have an accuracy operation ready to assess the
performance of our neural network.
'''
# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


'''
------------------------------------
# 3.2 Setting up the training
------------------------------------
'''

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))



'''
--------------------------------------------------------------------------------
# Word 2 Vec
--------------------------------------------------------------------------------
http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
'''
import os
import urllib
import zipfile

def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


url = 'http://mattmahoney.net/dc/'
filename = maybe_download('text8.zip', url, 31344016)

# Read the data into a list of strings. (From Zip)
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
print(vocabulary[:7])
#['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']



def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


import collections

# data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


#
import nltk
word2idx = { 'PAD': 0, 'hello':1, 'world':2 }
features = {}
features['word_indices'] = nltk.word_tokenize('hello world') # ['hello', 'world']
features['word_indices'] = [word2idx.get(word, UNKNOWN_TOKEN) for word in features['word_indices']]
UNKNOWN_TOKEN=len(weights)
word2idx.get('PAD')



#
