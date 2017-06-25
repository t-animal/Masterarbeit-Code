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
from util.plot import plotPCA, plotLDA, plotHistogram, plotVectorList, plotPairDistances
from util.containers import LazyModel, TestresultContainer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
from sklearn.utils.validation import check_X_y, check_is_fitted

log = logging.getLogger("de.t_animal.MA.util.classifiers")


class Classifier(BaseEstimator, ClassifierMixin):
	""" A classifier base class. It mainly just specifies the interface """

	""" Stopwords are the 100 most frequent words from the wikipedia, except some words which we think might have an impact on detection rate"""
	stopwords = {'the', 'of', 'and', 'in', 'to', 'was', 'is', 'for', 'on', 'as', 'by', 'with',
	             'he', 'at', 'from', 'that', 'his', 'it', 'an', 'were', 'are', 'also', 'which',
	             'this', 'or', 'be', 'first', 'new', 'has', 'had', 'one', 'their', 'after', 'not',
	             'who', 'its', 'but', 'two', 'her', 'they', 'she', 'references', 'have', 'th', 'all',
	             'other', 'been', 'time', 'when', 'school', 'during', 'may', 'year', 'into', 'there',
	             'world', 'city', 'up', 'de', 'university', 'no', 'state', 'more', 'national', 'years',
	             'united', 'external', 'over', 'links', 'only', 'american', 'most', 'team', 'three',
	             'film', 'out', 'between', 'would', 'later', 'where', 'can', 'some', 'st', 'season',
	             'about', 'south', 'born', 'used', 'states', 'under', 'him', 'then', 'second', 'part',
	             'such', 'made', 'john', 'war', 'known', 'while'} - \
	             {'he', 'his', 'first', 'new', 'her', 'they', 'she', 'city', 'world', 'university', 'united',
	             'him', 'second'}

	def __init__(self):
		self.removedStopWords = {k: 0 for k in self.stopwords}

	def fit(self, X, y):
		"""Fit method as required by the scikit-learn estimator interface"""

		# import pdb
		# pdb.set_trace()
		# X, y = check_X_y(X, y)
		self.X_ = X
		self.y_ = y

		# Our classifiers determine the content and class of documents from filenames, so ignore y
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

	def plot(self, filenames, plotFunc = None):
		"""Abstract method, do your magic here.
		   It should plot the given filenames in a suitable manner.
		   Can be passed an additional parameter if more than one plotting function is avaialable"""
		raise NotImplemented()

	def _getDescribingVectors(self, filename):
		"""Returns a list of vectors describing the given file, for when a
		   generator is not useful"""
		return [x for x in self._generateDescribingVectors(filename)]


	def _generateDescribingVectors(self, filename):
		"""Abstract method, do your magic here.
		   A generator creating the vectors describing the given file
		"""
		raise NotImplemented()

	def removeStopWords(self, vectorTuples):
		"""Removes stopwords from an iterable of (word, vector)-tuples. It counts which word was removed how often
		   and this counter can be returned by getStopWordCounter. Not threadsafe!"""
		self.removedStopWords = {k: 0 for k in self.stopwords}
		for vectorTuple in vectorTuples:
			if vectorTuple[0] in self.stopwords:
				log.debug("Not counting stopword %s", vectorTuple[0])
				self.removedStopWords[vectorTuple[0]] += 1
				continue
			yield vectorTuple

	def getStopWordCounter(self):
		return self.removedStopWords


class EmbeddingsClassifier(Classifier):
	""" An embeddings classifier base class: It vectorizes a file.
	"""

	def __init__(self, modelPath = None):
		if modelPath is None:
			raise ValueError("Model Path may not be None") #this breaks scikit-learn api

		self.modelPath = modelPath
		self.model = LazyModel(KeyedVectors.load_word2vec_format, modelPath, binary=modelPath.endswith("bin"))


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

	def plot(self, filenames, plotFunc="PCA"):
		"""Plots document vectors of the supplied files. Thi function can
		   plot the PCA and LDA (each 2D and 3D), the histogram of the
		   vectors and the vectors as a matrix

		   :param filenames: the files to plot
		   :type filenames: iterable of strings

		   :param plotFunc: the plotting function name to use
		   :type filenames: one of ["PCA", "PCA3", "LDA", "LDA3" "HIST", "MAT", "DIST"]
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
		if plotFunc == "DIST":
			plotPairDistances(arousedVectors, nonArousedVectors)


class SVMClassifierMixin:
	""" A mixin providing a train-method for SVMs. It iterates over
		files and their vectors and trains an SVM with them. Implement
		testing on your own. """

	def trainSVM(self, trainFilenames, svmParams={}):
		"""Trains an SVM using the files supplied in trainFilenames. The dict svmParams is passed
		   as keyword parameters to the svm. By default the svm parameter, class_weights will be
		   set to "balanced" if it has not been set explicitly. The svm paramater random_state
		   will be overwritten and set to a fixed value.

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		   :param svmParams: values to pass to the svm as keyword arguments
		   :type svmParams: dict
		"""
		log.info("Training SVM from scratch, collecting document vectors first")
		nonArousedVectors = []
		arousedVectors = []

		for filename in trainFilenames:
			log.info("Beginning with file %s", filename)
			vectors = self._generateDescribingVectors(filename)

			for vector in vectors:
				if isAroused(filename):
					arousedVectors.append(vector.flatten())
				else:
					nonArousedVectors.append(vector.flatten())

				log.debug("Vector: %s", str(vector))

		log.info("Start training svm with document vectors")

		if "class_weight" not in svmParams:
			svmParams["class_weight"] = "balanced"

		svmParams["random_state"] = 42
		self.svm = SVM.SVC(**svmParams) # TODO: Plot results for weights
		self.svm.fit(nonArousedVectors + arousedVectors,
		             [0] * len(nonArousedVectors) + [1] * len(arousedVectors))

	def testSVM(self, testFilenames):
		"""Test the SVM using the files supplied in testFilenames. It's very simple
		   and assumes a file can be described using a single feature vector.

		   :param testFilenames: The filenames to test upon
		   :type testFilenames: iterable of strings
		   :returns: a TestresultContainer object
		   :raises RuntimeError: if no svm was trained or loaded
		"""
		if self.svm is None:
			raise RuntimeError("Call train or load before calling test!")

		log.info("Beginning testing")
		testResult = TestresultContainer(True, False, "aroused", "nonAroused")

		for filename in testFilenames:
			feature = self._getDescribingVectors(filename)[0].flatten()

			result = self.svm.predict([feature])[0]
			testResult.addResult(bool(result), isAroused(filename))

			log.info("Checked file %s, result %s (%s)", filename, result,
			         "CORRECT" if bool(result) == isAroused(filename) else "INCORRECT")

		log.info("Finished testing: %s", testResult.oneline())

		return testResult

	def loadSVM(self, svmPath):
		"""Load the internal state from a path
		"""
		log.info("Loading SVM from pickled file")
		with open(svmPath, "rb") as persistanceFile:
			self.svm = pickle.load(persistanceFile, encoding='latin1')


	def storeSVM(self, svmPath):
		"""Store the internal state to a path
		"""
		if self.svm is None:
			raise RuntimeError("Call train or load before calling save!")

		log.info("Storing SVM to file")
		with open(svmPath, "wb") as persistanceFile:
			pickle.dump(self.svm, persistanceFile)


class RandomForestClassifierMixin:
	""" A mixin providing a train-method for Random Forests. It iterates over
		files and their vectors and trains a random forest classifier with them. Implement
		testing on your own if you need fancy stuff. """

	def trainForests(self, trainFilenames, forestParams={}):
		"""Trains random forests using the files supplied in trainFilenames. The dict forestParams is passed
		   as keyword parameters to the classifier. By default the classifier parameter, class_weights will be
		   set to "balanced" if it has not been set explicitly. The classifier paramater random_state
		   will be overwritten and set to a fixed value.

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		   :param forestParams: values to pass to the classifier as keyword arguments
		   :type forestParams: dict
		"""
		log.info("Training forests from scratch, collecting document vectors first")
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

		log.info("Start training forests with document vectors")

		if "class_weight" not in forestParams:
			forestParams["class_weight"] = "balanced"

		forestParams["random_state"] = 42
		self.forests = ensemble.RandomForestClassifier(**forestParams) # TODO: Plot results for weights
		self.forests.fit(nonArousedVectors + arousedVectors,
		             [0] * len(nonArousedVectors) + [1] * len(arousedVectors))

	def testForests(self, testFilenames):
		"""Test the forests using the files supplied in testFilenames. It's very simple
		   and assumes a file can be described using a single feature vector.

		   :param testFilenames: The filenames to test upon
		   :type testFilenames: iterable of strings
		   :returns: a TestresultContainer object
		   :raises RuntimeError: if no svm was trained or loaded
		"""
		if self.forests is None:
			raise RuntimeError("Call train or load before calling test!")

		log.info("Beginning testing")
		testResult = TestresultContainer(True, False, "aroused", "nonAroused")

		for filename in testFilenames:
			feature = self._getDescribingVectors(filename)[0]

			result = self.forests.predict([feature])[0]
			testResult.addResult(bool(result), isAroused(filename))

			log.info("Checked file %s, result %s (%s)", filename, result,
			         "CORRECT" if bool(result) == isAroused(filename) else "INCORRECT")

		log.info("Finished testing: %s", testResult.oneline())

		return testResult

	def loadForests(self, forestsPath):
		"""Load the internal state from a path
		"""
		log.info("Loading forests from pickled file")
		with open(forestsPath, "rb") as persistanceFile:
			self.forests = pickle.load(persistanceFile, encoding='latin1')


	def storeForests(self, forestsPath):
		"""Store the internal state to a path
		"""
		if self.forests is None:
			raise RuntimeError("Call train or load before calling save!")

		log.info("Storing forests to file")
		with open(forestsPath, "wb") as persistanceFile:
			pickle.dump(self.forests, persistanceFile)