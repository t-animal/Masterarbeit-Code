#!./venv/bin/python

import logging as log
import numpy as np
import os
import pdb
import pickle
import re

from sklearn.preprocessing import normalize
from util.classifiers import EmbeddingsClassifier, RandomForestClassifierMixin

class DocSumRandomForestsClassifier(RandomForestClassifierMixin, EmbeddingsClassifier):
	""" A very simple classifier: It vectorizes a file, averages up all word vectors
	    and trains a random forest with the result
	"""

	def __init__(self, modelPath = None, estimators = 30, max_features = "sqrt", nostopwords = False, norm="l2", useImageID = False):
		super().__init__(modelPath)
		self.estimators = int(estimators)
		self.max_features = max_features if max_features in ["auto", "sqrt", "log"] else float(max_features)
		self.bootstrap = True

		self.norm = norm
		self.nostopwords = nostopwords
		self.useImageID = useImageID

	def _generateDescribingVectors(self, filename):
		"""Vectorizes a file, averages the vectors

		   :param filename: the file to process
		   :type filename: string
		   :returns: numpy array
		"""
		if self.nostopwords:
			vectorizedFile = list(self.removeStopWords(self._vectorizeFile(filename)))
		else:
			vectorizedFile = self._vectorizeFile(filename)

		fileSum = np.zeros(vectorizedFile[0][1].size, dtype=np.float64).reshape(1,-1)
		for token, vector in vectorizedFile:
			fileSum += vector

		if self.norm == "length":
			fileSum /= len(vectorizedFile)
		else:
			fileSum = normalize(fileSum, self.norm)

		if self.nostopwords:
			stopWordCounter = np.array([[v for k,v in sorted(self.getStopWordCounter().items())]], dtype=np.float64)
			if self.norm == "length":
				stopWordCounter /= len(vectorizedFile)
			else:
				stopWordCounter = normalize(stopWordCounter, self.norm)

			fileSum = np.append(fileSum, stopWordCounter).reshape(1,-1)

		if self.useImageID:
			fileSum = np.append(fileSum, [getImageID(filename)])

		yield fileSum


	def train(self, trainFilenames):
		self.trainForests(trainFilenames, {"n_estimators": self.estimators,
		                                  "bootstrap": self.bootstrap,
		                                  "class_weight": "balanced"})

	test = RandomForestClassifierMixin.testForests
