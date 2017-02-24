#!./venv/bin/python

import logging as log
import numpy as np
import os
import pdb
import pickle
import re

from sklearn import svm as SVM
from gensim.models import Word2Vec
from util import softmax, isAroused
from util.plot import plotPCA
from util.LazyModel import LazyModel

class DocSumSVMClassifier():
	""" A very simple classifier: It vectorizes a file, averages up all word vectors
	    and trains an SVM with the result
	"""

	def __init__(self, modelPath):
		self.modelPath = modelPath
		self.model = LazyModel(Word2Vec.load_word2vec_format, modelPath, binary=True)
		self.svm = None


	def _getDocumentSum(self, filename):
		"""Vectorizes a file, averages the vectors

		   :param filename: the file to process
		   :type filename: string
		   :returns: numpy array
		"""
		vectorizedFile = self._vectorizeFile(filename)

		fileSum = np.zeros(300, dtype=np.float32)
		for token, vector in vectorizedFile:
			fileSum += vector

		fileSum **= 3
		fileSum /= len(vectorizedFile)

		return fileSum


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
			fileSum = self._getDocumentSum(filename)

			if isAroused(filename):
				arousedVectors.append(fileSum)
			else:
				nonArousedVectors.append(fileSum)

			log.debug("Vector sum: %s", str(fileSum))

		log.info("Start training svm with document sums")
		self.svm = SVM.SVC(probability = True)
		self.svm.fit(nonArousedVectors + arousedVectors,
		             [0] * len(nonArousedVectors) + [1] * len(arousedVectors))


	def test(self, testFilenames):
		"""Test the SVM using the files supplied in testFilenames.

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		   :returns: a results dict
		   :raises RuntimeError: if no svm was trained or loaded
		"""
		if self.svm is None:
			raise RuntimeError("Call train or load before calling test!")

		log.info("Beginning testing")
		correct = 0
		correctAroused = 0
		totalAroused = 0
		distances = {}
		for filename in testFilenames:
			fileSum = self._getDocumentSum(filename)

			result = self.svm.predict([fileSum])[0]
			if bool(result) == isAroused(filename):
				correct += 1
				if isAroused(filename):
					correctAroused += 1

			if isAroused(filename):
				totalAroused += 1

			distance = self.svm.decision_function([fileSum])[0]
			distances[filename] = distance

			log.info("Checked file %s, result %s (%s), distance: %s", filename, result,
			         "CORRECT" if bool(result) == isAroused(filename) else "INCORRECT", distance)

		log.info("Finished testing: %s correct out of %s (%s%%)", correct, len(testFilenames), float(correct)/len(testFilenames)*100)

		return {
			"tested": len(testFilenames),
			"correct": correct,
			"correct-percentage": float(correct)/len(testFilenames)*100,
			"correct-aroused": correctAroused,
			"correct-nonaroused": correct - correctAroused,
			"correct-aroused-percentage": float(correctAroused)/totalAroused*100,
			"correct-nonaroused-percentage": float(correct - correctAroused)/(len(testFilenames)-totalAroused)*100,
			"incorrect": len(testFilenames) - correct,
			"incorrect-aroused": totalAroused - correctAroused,
			"incorrect-nonaroused": len(testFilenames) - correct - (totalAroused - correctAroused),
			"additional": {"distances": distances}
		}


	def plot(self, filenames):
		"""Plots the PCA of the document sum of the supplied files

		   :param filenames: the files to plot
		   :type filenames: iterable of strings
		"""
		nonArousedVectors = []
		arousedVectors = []

		for filename in filenames:
			fileSum = self._getDocumentSum(filename)

			if isAroused(filename):
				arousedVectors.append(fileSum)
			else:
				nonArousedVectors.append(fileSum)

		plotPCA(nonArousedVectors, arousedVectors, "non aroused", "aroused")

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


	def load(self, svmPath):
		log.info("Loading SVM from pickled file")
		with open(svmPath, "rb") as persistanceFile:
			self.svm = pickle.load(persistanceFile, encoding='latin1')


	def save(self, svmPath):
		if self.svm is None:
			raise RuntimeError("Call train or load before calling save!")

		log.info("Storing SVM to file")
		with open(svmPath, "wb") as persistanceFile:
			pickle.dump(self.svm, persistanceFile)


if __name__ == "__main__":

	import argparse
	import json

	parser = argparse.ArgumentParser(description='Vectorize documents, sum the vectors, pass to SVM for classification')
	parser.add_argument("-v", help="Be more verbose (-vv for max verbosity)", action="count", default=0)
	parser.add_argument("--human", help="Print human-readble output", action="store_true")
	parser.add_argument("--modelPath", help="Path to word2vec model", required=True)
	parser.add_argument("--train", help="Path to document(s) to train from", nargs="+", required=True)
	parser.add_argument("--test", help="Path to document(s) to test against", nargs="+",
	                    required=True)

	args = parser.parse_args()

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	try:
		classifier = DocSumSVMClassifier(args.modelPath)

		if args.train:
			classifier.train(args.train)

		result = classifier.test(args.test)
		if args.human:
			del result["additional"]
			print(json.dumps(result, indent=1))
		else:
			print(json.dumps(result))
	except KeyboardInterrupt:
		pass
