import gensim
import logging
import numpy as np
import os
import pdb
import pickle
import re

from sklearn import svm as SVM
from gensim.models import KeyedVectors
from util import softmax, isAroused
from util.plot import plotPCA, plotLDA, plotHistogram, plotVectorList
from util.containers import LazyModel, TestresultContainer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

log = logging.getLogger("de.t_animal.MA.util.classifiers")

class Classifier(BaseEstimator, ClassifierMixin):
	""" A classifier base class: It vectorizes a file.
	"""

	def __init__(self, modelPath = None):
		if modelPath is None:
			raise ValueError("Model Path may not be None") #this breaks scikit-learn api

		self.modelPath = modelPath
		self.model = LazyModel(KeyedVectors.load_word2vec_format, modelPath, binary=modelPath.endswith("bin"))


	def _getDescribingVectors(self, filename):
		"""Returns a list of vectors describing the given file, for when a
		   generator is not useful"""
		return [x for x in self._generateDescribingVectors(filename)]


	def _generateDescribingVectors(self, filename):
		"""Abstract method, do your magic here.
		   A generator creating the vectors describing the given file
		"""
		raise NotImplemented()

	def fit(self, X, y):
		"""Fit method as required by the scikit-learn estimator interface"""

		# import pdb
		# pdb.set_trace()
		# X, y = check_X_y(X, y)
		self.X_ = X
		self.y_ = y

		self.train(X)

		return self

	def predict(self, X):
		"""Predict method as required by the scikit-learn predictor interface"""

		check_is_fitted(self, ['X_', 'y_'])

		raise NotImplemented()

	def train(self, trainFilenames):
		"""Abstract method, do your magic here.
		   It should train the classifier with the supplied filenames.

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		"""
		raise NotImplemented()


	def test(self, testFilenames):
		"""Abstract method, do your magic here.
		   It should test the classifier with the supplied filenames.

		   :param testFilenames: The filenames to test
		   :type testFilenames: iterable of strings
		"""
		raise NotImplemented()


	def _vectorizeFile(self, filename):
		""" Opens a file, splits it into words and gets the vectors for each word
		    that is contained in the model. It caches the results so that subsequent
		    runs can be faster.

		    :param filename: the filename of the file to vectorize
		    :type filename: string
		    :return: a list of tuples (word, vector)
		 """

		cachePath = os.path.split(filename)
		cachePath = os.path.join(cachePath[0], "." + cachePath[1] + ".veccache")
		cache = {}
		try:
			with open(cachePath, "rb") as cacheFile:
				cache = pickle.load(cacheFile, encoding='latin1')
				translatedFile = cache[os.path.realpath(self.modelPath)]
				log.debug("loaded vectorized file %s from cache", filename)

		except Exception:
			translatedFile = []

			with open(filename, "r") as file:
				content = file.read()

				for token in gensim.utils.simple_preprocess(content):
					try:
						translatedFile.append((token, self.model[token]))
					except KeyError:
						log.warn("token '%s' not in vocabulary", token)
			try:
				with open(cachePath, "wb") as cacheFile:
					log.debug("storing vectorized file %s to cache", filename)
					cache[self.modelPath] = translatedFile
					pickle.dump(cache, cacheFile)
			except Exception as e:
				log.warn("Could not store cache: %s", str(e))

		return translatedFile


	def load(self, path):
		"""Abstract method, do your magic here.
		   It should load the internal state from a path
		"""
		raise NotImplemented()

	def store(self, path):
		"""Abstract method, do your magic here.
		   It should store the internal state to a path
		"""
		raise NotImplemented()

	def plot(self, filenames, plotFunc="PCA"):
		"""Plots document vectors of the supplied files. Thi function can
		   plot the PCA and LDA (each 2D and 3D), the histogram of the
		   vectors and the vectors as a matrix

		   :param filenames: the files to plot
		   :type filenames: iterable of strings

		   :param plotFunc: the plotting function name to use
		   :type filenames: one of ["PCA", "PCA3", "LDA", "LDA3" "HIST", "MAT"]
		"""
		nonArousedVectors = []
		arousedVectors = []

		for filename in filenames:
			fileSum = self._getDescribingVectors(filename)[0]

			if isAroused(filename):
				arousedVectors.append(fileSum)
			else:
				nonArousedVectors.append(fileSum)

		if plotFunc == "PCA":
			plotPCA(nonArousedVectors, arousedVectors, "non aroused", "aroused")
		if plotFunc == "PCA3":
			plotPCA(nonArousedVectors, arousedVectors, "non aroused", "aroused", 3)
		if plotFunc == "LDA":
			plotLDA(nonArousedVectors, arousedVectors, "non aroused", "aroused")
		if plotFunc == "LDA3":
			plotLDA(nonArousedVectors, arousedVectors, "non aroused", "aroused", 3)
		if plotFunc == "MAT":
			plotVectorList(arousedVectors, nonArousedVectors)
		if plotFunc == "HIST":
			plotHistogram(arousedVectors, nonArousedVectors)


class SVMClassifier(Classifier):

	def __init__(self, modelPath):
		super().__init__(modelPath)


	def train(self, trainFilenames, svmParams={}):
		"""Trains an SVM using the files supplied in trainFilenames.

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		   :param svmParams: values to pass to the svm as keyword arguments
		   :type svmParams: dict
		"""
		log.info("Training SVM from scratch, collecting document sums first")
		nonArousedVectors = []
		arousedVectors = []

		for filename in trainFilenames:
			log.info("Beginning with file %s", filename)
			vectors = self._generateDescribingVectors(filename)

			for vector in vectors:
				if isAroused(filename):
					arousedVectors.append(vector)
				else:
					nonArousedVectors.append(vector)

				log.debug("Vector: %s", str(vector))

		log.info("Start training svm with document vectors")
		self.svm = SVM.SVC(**svmParams) # TODO: Plot results for weights
		self.svm.fit(nonArousedVectors + arousedVectors,
		             [0] * len(nonArousedVectors) + [1] * len(arousedVectors))


	def load(self, svmPath):
		"""Load the internal state from a path
		"""
		log.info("Loading SVM from pickled file")
		with open(svmPath, "rb") as persistanceFile:
			self.svm = pickle.load(persistanceFile, encoding='latin1')


	def store(self, svmPath):
		"""Store the internal state to a path
		"""
		if self.svm is None:
			raise RuntimeError("Call train or load before calling save!")

		log.info("Storing SVM to file")
		with open(svmPath, "wb") as persistanceFile:
			pickle.dump(self.svm, persistanceFile)