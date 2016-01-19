#!/usr/bin/python
#-*- coding: utf-8 -*-
### gender predictor ###
######### May 21, 2015 ########
########## Mindy Foster ##########


from __future__ import division
import numpy as np
import gensim
import cPickle as pickle
import NNet
reload(NNet)
from NNet import NeuralNet



def gp(names):
	model = gensim.models.Word2Vec.load('gmodel_en')
	nn = len(model['a'])
	with open('gender_classifier.db','rb') as infile:
		nnet = pickle.load(infile)
	out = []
	for name in names:
		for letter in name:
			try:
				vec = np.zeros(nn).reshape((1,nn))
				vec += model[letter.lower()].reshape((1,nn))
			except Exception:
				continue
		prop = nnet.predict_proba(vec)[:,1]
		if prop>0 and prop < .16:
			out.append('M')
		elif prop >= .16:
			out.append('F')
	return out