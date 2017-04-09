#!./venv/bin/python

import logging as log
import numpy as np
import pdb

from sklearn import svm as SVM
from util import isAroused, softmax
from util.containers import TestresultContainer
from util.classifiers import EmbeddingsClassifier, SVMClassifierMixin
from util.plot import plotPCA

class NGramSVM(EmbeddingsClassifier, SVMClassifierMixin):

	def __init__(self, modelPath, windowSize, arousalLimit=0):
		# this calls the EmbeddingsClassifier
		super().__init__(modelPath)
		self.windowSize = int(windowSize)
		self.arousalLimit = float(arousalLimit)

	def _generateDescribingVectors(self, filename):
		"""Vectorizes a file, averages the vectors

		   :param filename: the file to process
		   :type filename: string
		   :returns: numpy array
		"""
		halfWindow = int(self.windowSize/2)
		vectorizedFile = self._vectorizeFile(filename)

		if len(vectorizedFile) <= self.windowSize:
			log.warning("Could not calculate vector of %s (windowSize > word count)", filename)

		for index in range(halfWindow, len(vectorizedFile) - halfWindow):
			tmpSum = np.zeros(vectorizedFile[0][1].size, dtype=np.float64)

			if len(vectorizedFile) <= self.windowSize:
				continue

			for offset in range(-halfWindow, halfWindow):
				tmpSum += vectorizedFile[index + offset][1]

			tmpSum **= 2
			tmpSum /= self.windowSize

			yield tmpSum

	def train(self, trainFilenames):
		# this calls the SVMClassifierMixin
		super().train(trainFilenames, {"probability": False, "random_state": 42, "class_weight": {True: 1, False: 3}})
		return

		log.info("Finished training, determining suggested threshold")

		arousalLimit = 0
		for filename in trainFilenames:
			fileSums = self._generateDescribingVectors(filename)

			totalAroused = 0
			totalNonAroused = 0

			for fileSum in fileSums:

				if bool(self.svm.predict([fileSum])[0]):
					totalAroused += 1
				else:
					totalNonAroused += 1

			if totalNonAroused == 0:
				continue

			arousalLimit += totalAroused/totalNonAroused

		arousalLimit /= len(trainFilenames)

		print("Suggested limit: " + str(arousalLimit))


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
			fileSums = self._generateDescribingVectors(filename)

			totalAroused = 0
			totalNonAroused = 0
			totalArousedDistance = 0
			totalNonArousedDistance = 0

			for fileSum in fileSums:

				distance = self.svm.decision_function([fileSum])[0]
				testResult.additional(distances = {filename + "-" + str(totalAroused+totalNonAroused): distance})


				if any(self.svm.predict_proba([fileSum])[0] > 0.7):

					if bool(self.svm.predict([fileSum])[0]):
						totalAroused += 1
						totalArousedDistance += abs(distance)
					else:
						totalNonAroused += 1
						totalNonArousedDistance += abs(distance)

			testResult.additional(classifyCount = {filename: {"isAroused": isAroused(filename), "aroused": totalAroused, "nonAroused": totalNonAroused}})

			if totalNonAroused == 0:
				testResult.addResult(True, isAroused(filename))
				continue

			if totalAroused == 0:
				testResult.addResult(False, isAroused(filename))
				continue

			# testResult.addResult(totalAroused/totalNonAroused >= self.arousalLimit, isAroused(filename))
			testResult.addResult(totalArousedDistance/totalAroused > totalNonArousedDistance/totalNonAroused, isAroused(filename))

			log.info("Checked file %s, aroused: %d (%d) non-aroused: %d (%d)", filename, totalAroused, totalArousedDistance, totalNonAroused, totalNonArousedDistance)

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

			for vector in self._generateDescribingVectors(filename):

				if isAroused(filename):
					arousedVectors.append(vector)
				else:
					nonArousedVectors.append(vector)

		plotPCA(nonArousedVectors, arousedVectors, "non aroused", "aroused")


if __name__ == "__main__":

	import argparse, os, sys

	parser = argparse.ArgumentParser(description='Vectorize documents, sum the vectors, pass to SVM for classification')
	parser.add_argument("-v", help="Be more verbose (-vv for max verbosity)", action="count", default=0)
	parser.add_argument("--human", help="Print human-readble output", action="store_true")
	parser.add_argument("--modelPath", help="Path to word2vec model", required=True)
	parser.add_argument("--windowSize", help="The windowsize to use for the ngram", type=int, required=True)
	parser.add_argument("--threshold", help="The threshold of aroused/nonaroused ratio", type=float, required=True)
	parser.add_argument("--plot", help="Plot the PCA of the document sum of files", nargs="+")
	parser.add_argument("--train", help="Path to document(s) to train from", nargs="+")
	parser.add_argument("--test", help="Path to document(s) to test against", nargs="+")

	args = parser.parse_args()

	if not args.plot and ( not args.test or not args.train ):
		parser.error("Supply either test and train or plot")

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	try:
		classifier = NGramSVM(args.modelPath, args.windowSize, args.threshold)

		if args.train:
			classifier.train(args.train)

		if args.test:
			result = classifier.test(args.test)

			if args.human:
				if os.isatty(sys.stdout.fileno()):
					print("\033[1m"+result.oneline()+"\033[0m")
				else:
					print(result.oneline())
				print(result)
			else:
				print(result.getJSON())

		if args.plot:
			classifier.plot(args.plot)

	except KeyboardInterrupt:
		pass
