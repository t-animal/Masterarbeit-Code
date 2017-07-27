#!./venv/bin/python

import logging as log
import numpy as np
import os
import pdb
import pickle
import random
import re
import warnings

from scipy.cluster.vq import whiten, kmeans, kmeans2
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from gensim.models import Word2Vec
from util import softmax, isAroused
from util.plot import plotPCA
from util.containers import LazyModel, TestresultContainer
from util.classifiers import EmbeddingsClassifier, SVMClassifierMixin

class KMeansSVMClassifier(SVMClassifierMixin, EmbeddingsClassifier):
	""" A very simple classifier: It vectorizes a file, averages up all word vectors
	    and trains an SVM with the result
	"""

	def __init__(self, modelPath = None, k = 5, SVM_C = 2.5, gamma = "auto", nostopwords = False, norm="l2"):
		super().__init__(modelPath)
		self.k = int(k)
		self.SVM_C = float(SVM_C)
		self.gamma = "auto" if gamma == "auto" else float(gamma)
		self.nostopwords = nostopwords
		self.norm = norm

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

		observations = whiten(np.array([v for t,v in vectorizedFile]))

		result, indices = kmeans2(observations, self.kMeansSeed, check_finite = False)

		# calculate the result by ourself so that we can control the norm applied
		binSizes = [0] * self.k
		bins = [None] * self.k
		for vectorIndex, binIndex in enumerate(indices):
			if bins[binIndex] is not None:
				bins[binIndex] += vectorizedFile[vectorIndex][1]
			else:
				bins[binIndex] = vectorizedFile[vectorIndex][1]

			binSizes[binIndex] += 1


		if any(b is None for b in bins):
			log.info("At least one bin is empty! Setting zeros instead...")
			bins = [binSum if binSum is not None else np.zeros(vectorizedFile[0][1].shape) for binSum in bins]
			binSizes = [max(1, s) for s in binSizes]

		if self.norm == "length":
			result = [binSum / length for binSum, length in zip(bins, binSizes)]
		else:
			result = [normalize(binSum.reshape(1,-1), self.norm) for binSum in bins]

		result = np.concatenate(result)

		if self.nostopwords:
			stopWordCounter = np.array([[v for k,v in sorted(self.getStopWordCounter().items())]], dtype=np.float64)
			if self.norm == "length":
				stopWordCounter /= len(vectorizedFile)
			else:
				stopWordCounter = normalize(stopWordCounter, self.norm)

			result = np.append(result, stopWordCounter)


		yield result.reshape(1,-1)

	def train(self, trainFilenames):
		vectors = []
		for filename in trainFilenames:
			if self.nostopwords:
				vectors += [v for t,v in self.removeStopWords(self._vectorizeFile(filename))]
			else:
				vectors += [v for t,v in self._vectorizeFile(filename)]

		random.seed(23)
		result, distortion = None, 1e42
		bestIteration = 0
		for i in range(30):
			curResult = []
			while not len(curResult) == self.k:
				initial = np.array(vectors)[random.sample(range(len(vectors)), self.k)]
				curResult, curDistortion = kmeans(vectors, initial, check_finite = False)

			if curDistortion < distortion:
				bestIteration = i
				result, distortion = curResult, curDistortion
		print("No improvement after iteration no " + str(bestIteration))

		self.kMeansSeed = result

		self.trainSVM(trainFilenames, {"C": self.SVM_C,
		                               "gamma": self.gamma,
		                               "class_weight": "balanced"})


	test = SVMClassifierMixin.testSVM

	def predict(self, X):
		"""Predict method as required by the scikit-learn predictor interface"""

		check_is_fitted(self, ['X_', 'y_'])

		return self.svm.predict(list(map(lambda x: self._getDescribingVectors(x)[0], X)))


if __name__ == "__main__":

	import argparse, sys

	parser = argparse.ArgumentParser(description='Vectorize documents, sum the vectors, pass to SVM for classification')
	parser.add_argument("-v", help="Be more verbose (-vv for max verbosity)", action="count", default=0)
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
