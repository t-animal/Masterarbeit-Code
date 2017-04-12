#!./venv/bin/python

import logging as log

from util import isAroused
from util.classifiers import Classifier, SVMClassifierMixin
from util.containers import TestresultContainer

from sklearn.feature_extraction.text import TfidfVectorizer

class BagOfWordsClassifier(SVMClassifierMixin, Classifier):
	"""A bag of words classifiers"""

	def __init__(self, SVM_C = 2.5, gamma="auto", ngram_range=[1,1]):
		self.SVM_C = float(SVM_C)
		self.gamma = gamma if gamma == "auto" else float(gamma)
		self.ngram_range = [a for a in map(int, ngram_range[1:-1].split(","))] if type(ngram_range) == str else ngram_range

	def _generateDescribingVectors(self, filename):
		a= self.vectorizer.transform([filename]).getrow(0).toarray()[0]
		# import pdb
		# pdb.set_trace()
		yield a

	def train(self, trainFilenames):
		"""Trains the classifier with the given filenames

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		"""

		self.vectorizer = TfidfVectorizer(input="filename", ngram_range=self.ngram_range)
		self.vectorizer.fit(trainFilenames)

		#this calls SVMClassifierMixin
		super().train(trainFilenames, {"C": self.SVM_C,
		                               "gamma": self.gamma,
		                               "random_state": 42,
		                               "class_weight": "balanced"})