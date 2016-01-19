# gender-predict
Given a name, predict gender!

This program uses python and a word2vec neural network model to predict gender.

It is trained on census data from 1950 to 2010 -- this data can be found in the names folder.

# Requirements
Python 2.x

Gensim https://radimrehurek.com/gensim/

cPickle

Numpy

# Usage
With all files in the same folder:

gp(name)

name can be either a single name or a list of names