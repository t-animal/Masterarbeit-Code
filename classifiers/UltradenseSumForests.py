import logging as log
import numpy as np
import os
import pdb
import pickle
import re

from DocSumRandomForests import DocSumRandomForestsClassifier

class UltradenseSumRandomForests(DocSumRandomForestsClassifier):
	""" A very simple classifier: It vectorizes a file, averages up all word vectors
	    and trains an SVM with the result
	"""

	def __init__(self, modelPath = None, estimators = 30, max_features = "sqrt", nostopwords = False, norm="l2", useImageID = False, qInfoPath = None):
		super().__init__(modelPath, estimators, max_features, nostopwords, norm, useImageID)
		self.Q = None
		with open(qInfoPath, "rb") as qi:
			self.Qinfo = pickle.load(qi)

	def _generateDescribingVectors(self, filename):
		"""Vectorizes a file, averages the vectors

		   :param filename: the file to process
		   :type filename: string
		   :returns: numpy array
		"""
		for vector in super()._generateDescribingVectors(filename):
			if self.Q is None:
				self.Q = np.eye(vector.size)
				self.Q[0:self.Q.shape[0], 0:self.Q.shape[1]] = self.Qinfo["Q"]

			yield self.Q * vector