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
from util.containers import LazyModel, TestresultContainer
from util.classifiers import SVMClassifier

class DocSumSVMClassifier(SVMClassifier):
	""" A very simple classifier: It vectorizes a file, averages up all word vectors
	    and trains an SVM with the result
	"""

	def _generateDescribingVectors(self, filename):
		"""Vectorizes a file, averages the vectors

		   :param filename: the file to process
		   :type filename: string
		   :returns: numpy array
		"""
		vectorizedFile = self._vectorizeFile(filename)

		fileSum = np.zeros(vectorizedFile[0][1].size, dtype=np.float32)
		for token, vector in vectorizedFile:
			fileSum += vector

		fileSum **= 3
		fileSum /= len(vectorizedFile)

		yield fileSum


	def test(self, testFilenames):
		"""Test the SVM using the files supplied in testFilenames.

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		   :returns: a TestresultContainer object
		   :raises RuntimeError: if no svm was trained or loaded
		"""
		if self.svm is None:
			raise RuntimeError("Call train or load before calling test!")

		log.info("Beginning testing")
		testResult = TestresultContainer(True, False, "aroused", "nonAroused")

		for filename in testFilenames:
			fileSum = self._getDescribingVectors(filename)[0]

			result = self.svm.predict([fileSum])[0]
			testResult.addResult(bool(result), isAroused(filename))

			distance = self.svm.decision_function([fileSum])[0]
			testResult.additional(distances = {filename: distance})

			log.info("Checked file %s, result %s (%s), distance: %s", filename, result,
			         "CORRECT" if bool(result) == isAroused(filename) else "INCORRECT", distance)

		log.info("Finished testing: %s", testResult.oneline())

		return testResult


	def plot(self, filenames):
		"""Plots the PCA of the document sum of the supplied files

		   :param filenames: the files to plot
		   :type filenames: iterable of strings
		"""
		nonArousedVectors = []
		arousedVectors = []

		for filename in filenames:
			fileSum = self._getDescribingVectors(filename)[0]

			if isAroused(filename):
				arousedVectors.append(fileSum)
			else:
				nonArousedVectors.append(fileSum)

		plotPCA(nonArousedVectors, arousedVectors, "non aroused", "aroused")


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
