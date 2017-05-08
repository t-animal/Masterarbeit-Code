#!./venv/bin/python

import logging as log
import numpy as np
import os
import pdb
import pickle
import re

from sklearn import svm as SVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from gensim.models import Word2Vec
from util import softmax, isAroused
from util.plot import plotPCA
from util.containers import LazyModel, TestresultContainer
from util.classifiers import EmbeddingsClassifier, RandomForestClassifierMixin

class DocSumRandomForestsClassifier(RandomForestClassifierMixin, EmbeddingsClassifier):
	""" A very simple classifier: It vectorizes a file, averages up all word vectors
	    and trains an SVM with the result
	"""

	def __init__(self, modelPath = None, estimators = 30, max_features = "sqrt"):
		super().__init__(modelPath)
		self.estimators = int(estimators)
		self.max_features = max_features if max_features in ["auto", "sqrt", "log"] else float(max_features)
		self.bootstrap = True

	def _generateDescribingVectors(self, filename):
		"""Vectorizes a file, averages the vectors

		   :param filename: the file to process
		   :type filename: string
		   :returns: numpy array
		"""
		vectorizedFile = self._vectorizeFile(filename)

		fileSum = np.zeros(vectorizedFile[0][1].size, dtype=np.float64)
		for token, vector in vectorizedFile:
			fileSum += vector

		yield fileSum

	def train(self, trainFilenames):
		self.trainForests(trainFilenames, {"n_estimators": self.estimators,
		                                  "bootstrap": self.bootstrap,
		                                  "class_weight": "balanced"})

	test = RandomForestClassifierMixin.testForests

	def predict(self, X):
		"""Predict method as required by the scikit-learn predictor interface"""

		check_is_fitted(self, ['X_', 'y_'])

		return self.svm.predict(list(map(lambda x: self._getDescribingVectors(x)[0], X)))


if __name__ == "__main__":

	import argparse, sys

	parser = argparse.ArgumentParser(description='Vectorize documents, sum the vectors, pass to random forests for classification')
	parser.add_argument("-v", help="Be more verbose (repeat v for more verbosity)", action="count", default=0)
	parser.add_argument("--human", help="Print human-readble output", action="store_true")
	parser.add_argument("--modelPath", help="Path to word2vec model", required=True)
	parser.add_argument("--plot", help="Plot the PCA of the document sum of files", nargs="+")
	parser.add_argument("--train", help="Path to document(s) to train from", nargs="+")
	parser.add_argument("--test", help="Path to document(s) to test against", nargs="+")

	args = parser.parse_args()

	if not args.plot and ( not args.test or not args.train ):
		parser.error("Supply either test and train or plot")

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	try:
		classifier = DocSumSVMClassifier(args.modelPath)

		if args.train:
			classifier.train(args.train)

		if args.test:
			result = classifier.test(args.test)

		if args.plot:
			classifier.plot(args.plot)

		if args.human:
			if os.isatty(sys.stdout.fileno()):
				print("\033[1m"+result.oneline()+"\033[0m")
			else:
				print(result.oneline())
			print(result)
		else:
			print(result.getJSON())


	except KeyboardInterrupt:
		pass
