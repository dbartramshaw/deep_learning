#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict


###########################################################################################
# Intro to Word2Vec  - Concept
###########################################################################################
# http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

# Words closeby as classed as similar
words = ['queen', 'book', 'king', 'magazine', 'car', 'bike']
vectors = np.array([[0.1,   0.3],  # queen
                    [-0.5, -0.1],  # book
                    [0.2,   0.2],  # king
                    [-0.3, -0.2],  # magazine
                    [-0.5,  0.4],  # car
                    [-0.45, 0.3]]) # bike

#plot the words
plt.plot(vectors[:,0], vectors[:,1], 'o')
plt.xlim(-0.6, 0.3)
plt.ylim(-0.3, 0.5)
for word, x, y in zip(words, vectors[:,0], vectors[:,1]):
    plt.annotate(word, (x, y), size=12)
zip(words, vectors[:,0], vectors[:,1])
plt.show()


s = pd.Series([0.1, 0.4, 0.01, 0.2, 0.05],index=["pumpkin", "shoe", "tree", "prince", "luck"])
s.plot(kind='bar')
plt.ylabel("$P(w|Cinderella)$")
plt.show()



###########################################################################################
# Intro to Word2Vec  - Actual
###########################################################################################

# example data
sentences = ['the king loves the queen', 'the queen loves the king',
             'the dwarf hates the king', 'the queen hates the dwarf',
             'the dwarf poisons the king', 'the dwarf poisons the queen']


def Vocabulary():
    dictionary = defaultdict()
    dictionary.default_factory = lambda: len(dictionary)
    return dictionary


def docs2bow(docs, dictionary):
    """Transforms a list of strings into a list of lists where
    each unique item is converted into a unique integer."""
    for doc in docs:
        yield [dictionary[word] for word in doc.split()]



vocabulary = Vocabulary()
sentences_bow = list(docs2bow(sentences, vocabulary))
sentences_bow

len(vocabulary)
V, N = len(vocabulary), 3
WI = (np.random.random((V, N)) - 0.5) / N
WO = (np.random.random((N, V)) - 0.5) / V


###########################################################################################
# We now construct the two matrices WW and W′W′
###########################################################################################

# Points in space are represented by continous embeddings of words#
# All vectors are initialised as random points in space
# We need to learn better positions

# Each row ii in WW corresponds to word ii and each column jj corresponds to the jjth dimension.
WI


# Notice that W′W′ isn't simply the transpose of WW but a different matrix:
WO



###########################################################################################
# computing the posterior probability of an output word given some input word
###########################################################################################

# we compute the distance between the input word dwarf and the output word hates:
print np.dot(WI[vocabulary['dwarf']], WO.T[vocabulary['hates']])

vocabulary['dwarf']
WI[vocabulary['dwarf']]
WO.T[vocabulary['hates']]

WI[4]
WO.T[5]


# Using SOFTMAX REGRESSION we compute the posterior probability
p_hates_dwarf = (np.exp(np.dot(WI[vocabulary['dwarf']], WO.T[vocabulary['hates']])) /
                       sum(np.exp(np.dot(WI[vocabulary['dwarf']], WO.T[vocabulary[w]]))
                           for w in vocabulary))
print p_hates_dwarf



###########################################################################################
# updating the hiddent-to-output layer weights
###########################################################################################


target_word = 'king'
input_word = 'queen'
learning_rate = 1.0

for word in vocabulary:
    p_word_queen = (np.exp(np.dot(WO.T[vocabulary[word]], WI[vocabulary[input_word]])) /
                    sum(np.exp(np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]]))
                        for w in vocabulary))
    t = 1 if word == target_word else 0
    error = t - p_word_queen
    WO.T[vocabulary[word]] = (WO.T[vocabulary[word]] - learning_rate *
                              error * WI[vocabulary[input_word]])
print WO

# Updating the input-to-hidden weights
WI[vocabulary[input_word]] = WI[vocabulary[input_word]] - learning_rate * WO.sum(1)

for word in vocabulary:
    p = (np.exp(np.dot(WO.T[vocabulary[word]], WI[vocabulary[input_word]])) /
         sum(np.exp(np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]]))
             for w in vocabulary))
    print word, p


# The above only assumes the context is a single word
# This allowed us to simply copy the input vector to the hidden layer.



###########################################################################################
# Multi word context
###########################################################################################

target_word = 'king'
context = ['queen', 'loves']

# if the context CC comprises multiple words,
# instead of copying the input vector we take the mean of their input vectors as our hidden layer:
h = (WI[vocabulary['queen']] + WI[vocabulary['loves']]) / 2

# Then we apply the hidden-to-output layer update
for word in vocabulary:
    p_word_context = (np.exp(np.dot(WO.T[vocabulary[word]], h)) /
                            sum(np.exp(np.dot(WO.T[vocabulary[w]], h)) for w in vocabulary))
    t = 1 if word == target_word else 0
    error = t - p_word_context
    WO.T[vocabulary[word]] = WO.T[vocabulary[word]] - learning_rate * error * h
print WO


# Finally we update the vector of each input word in the context:
for input_word in context:
    WI[vocabulary[input_word]] = (WI[vocabulary[input_word]] - (1. / len(context)) *
                                  learning_rate * WO.sum(1))

h = (WI[vocabulary['queen']] + WI[vocabulary['loves']]) / 2
for word in vocabulary:
    p = (np.exp(np.dot(WO.T[vocabulary[word]], h)) /
               sum(np.exp(np.dot(WO.T[vocabulary[w]], h)) for w in vocabulary))
    print word, p



###########################################################################################
# Paragraph Vector
###########################################################################################

# The Paragraph Vector model attempts to learn fixed-length continuous representations from variable-length pieces of text.


V, N, P = len(vocabulary), 3, 5
WI = (np.random.random((V, N)) - 0.5) / N
WO = (np.random.random((N, V)) - 0.5) / V
D =  (np.random.random((P, N)) - 0.5) / N

sentences = ['snowboarding is dangerous', 'skydiving is dangerous',
             'escargots are tasty to some people', 'everyone loves tasty food',
             'the minister has some dangerous ideas']


# We first convert the sentences into a vectorial BOW representation:
vocabulary = Vocabulary()
sentences_bow = list(docs2bow(sentences, vocabulary))
sentences_bow

# Next we compute the posterior probability for each word in the vocabulary given the concatenation and averaging of the first paragraph
# and the context word snowboarding.
# We compute the error and update the hidden-to-output layer weights.

target_word = 'dangerous'
h = (D[0] + WI[vocabulary['snowboarding']]) / 2
learning_rate = 1.0

for word in vocabulary:
    p = (np.exp(np.dot(WO.T[vocabulary[word]], h)) / sum(np.exp(np.dot(WO.T[vocabulary[w]], h)) for w in vocabulary))
    t = 1 if word == target_word else 0
    error = t - p
    WO.T[vocabulary[word]] = (WO.T[vocabulary[word]] - learning_rate * error * h)
print WO

# We backpropagate the error to the input-to-hidden layer as follows:
EH = WO.sum(1)
WI[vocabulary['snowboarding']] = WI[vocabulary['snowboarding']] - 0.5 * learning_rate * EH
D[0] = D[0] - 0.5 * learning_rate * EH
