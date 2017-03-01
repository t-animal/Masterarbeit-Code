import logging
import numpy as np
import os
import pdb
import pickle
import re

from sklearn import svm as SVM
from gensim.models import Word2Vec
from util import softmax, isAroused
from util.plot import plotPCA
from util.containers import LazyModel, TestresultContainer


log = logging.getLogger("de.t_animal.MA.util.classifiers")

class Classifier():
	""" A classifier base class: It vectorizes a file.
	"""

	def __init__(self, modelPath):
		self.modelPath = modelPath
		self.model = LazyModel(Word2Vec.load_word2vec_format, modelPath, binary=True)


	def _getDescribingVectors(self, filename):
		"""Returns a list of vectors describing the given file, for when a
		   generator is not useful"""
		return [x for x in self._generateDescribingVectors(filename)]


	def _generateDescribingVectors(self, filename):
		"""Abstract method, do your magic here.
		   A generator creating the vectors describing the given file
		"""
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
		try:
			with open(cachePath, "rb") as cacheFile:
				translatedFile = pickle.load(cacheFile, encoding='latin1')[os.path.realpath(self.modelPath)]
				log.debug("loaded vectorized file %s from cache", filename)

		except Exception:
			translatedFile = []

			with open(filename, "r") as file:
				content = file.read()
				content = re.sub('[.,:;!?…=\'"`´‘’“”„#%\\()/*+-]', ' ', content)

				for token in content.lower().split():
					if token not in self.model.wv.vocab.keys():
						log.debug("token '%s' not in vocabulary", token)
						continue

					translatedFile.append((token, self.model[token]))

			try:
				with open(cachePath, "wb") as cacheFile:
					log.debug("storing vectorized file %s to cache", filename)
					pickle.dump({os.path.realpath(self.modelPath): translatedFile}, cacheFile)
			except Exception as e:
				log.warn("Could not store cache: %s", str(e))

		return translatedFile


	def load(self, path):
		"""Abstract method, do your magic here.
		   It should load the internal state from a path
		"""


	def save(self, path):
		"""Abstract method, do your magic here.
		   It should store the internal state to a path
		"""



class SVMClassifier(Classifier):

	def __init__(self, modelPath):
		super().__init__(modelPath)


	def train(self, trainFilenames):
		"""Trains an SVM using the files supplied in trainFilenames.

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
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
		self.svm = SVM.SVC(probability = True, class_weight={0:1, 1:1.1}) # TODO: Plot results for weights
		self.svm.fit(nonArousedVectors + arousedVectors,
		             [0] * len(nonArousedVectors) + [1] * len(arousedVectors))

	def load(self, svmPath):
		"""Load the internal state from a path
		"""
		log.info("Loading SVM from pickled file")
		with open(svmPath, "rb") as persistanceFile:
			self.svm = pickle.load(persistanceFile, encoding='latin1')


	def save(self, svmPath):
		"""Store the internal state to a path
		"""
		if self.svm is None:
			raise RuntimeError("Call train or load before calling save!")

		log.info("Storing SVM to file")
		with open(svmPath, "wb") as persistanceFile:
			pickle.dump(self.svm, persistanceFile)