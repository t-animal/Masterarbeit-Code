#!./venv/bin/python

import numpy as np
import pdb
import pickle

from .DocSumSVM import DocSumSVMClassifier

class UltradenseSumSVM(DocSumSVMClassifier):
	""" A very simple classifier: It vectorizes a file, averages up all word vectors
	    and trains an SVM with the result
	"""

	def __init__(self, modelPath = None, SVM_C = 2.5, gamma = "auto", nostopwords = False, norm="l2", useImageID = False, qInfoPath = None):
		super().__init__(modelPath, SVM_C, gamma, nostopwords, norm, useImageID)
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