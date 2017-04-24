import itertools
import logging
import numpy as np
import operator
import os
import random

from math import ceil, floor

from util import isAroused
from util.containers import SubsetKeyedVectors, TestresultContainer
from util.classifiers import EmbeddingsClassifier, SVMClassifierMixin

from util.deBoomReprLearning import WeightLearningNetwork, getDocumentFrequencies

log = logging.getLogger("de.t_animal.MA.classifiers.WMDClassifier")

class DeBoomClassifier(SVMClassifierMixin, EmbeddingsClassifier):

	def __init__(self, modelPath):
		super().__init__(modelPath)

		self.subsetModel = SubsetKeyedVectors()
		self.weights = None
		self.docFreqs = None


	def _generateDescribingVectors(self, filename):
		if self.weights is None:
			raise ValueError("Call learnRepresentationWeights before using this method")

		vectorizedFile = [vecTuple for vecTuple in self._vectorizeFile(filename) if vecTuple[0] in self.subsetModel]
		vectorizedFile.sort(key = lambda vecTuple: self.docFreqs[self.subsetModel.vocab[vecTuple[0]].index])

		nmax = len(self.weights)
		nCl = len(vectorizedFile)
		if not nmax == nCl:
			#we add a list element to the front of lists to allow 1-based indexing like in the formulas in the paper
			w = [None] + list(self.weights)
			I = [None] + [1 + (j - 1) * (nmax - 1) / (nCl - 1) for j in range(1, nCl + 1)] #eq 12 in the paper
			e = 1e-10
			interpolatedWeights = [((w[ceil(I[j])] - w[floor(I[j])]) * (I[j] - floor(I[j])))
			                       / (ceil(I[j]) - floor(I[j]) + e) + w[ceil(I[j])] for j in range(1, nCl + 1)] #eq 13
		else:
			interpolatedWeights = self.weights

		print(interpolatedWeights)

		fileSum = np.zeros(vectorizedFile[0][1].size, dtype=np.float64)
		for index, (token, vec) in enumerate(vectorizedFile):
			try:
				fileSum += vec * interpolatedWeights[index]
			except IndexError:
				break

		fileSum /= len(vectorizedFile)

		yield fileSum


	def learnRepresentationWeights(self, trainFilenames):
		""" Learns the representation weights according to deBoom's paper
		    Representation learning for very short texts using weighted word embedding aggregation"""

		vectorizedFiles = list(map(self._vectorizeFile, trainFilenames))
		tokensPerFile = [[vecTuple[0] for vecTuple in vectorTuples] for vectorTuples in vectorizedFiles]
		maxLen = max(map(len, tokensPerFile))

		self.subsetModel.addToVocab(list(itertools.chain.from_iterable(vectorizedFiles)))
		self.docFreqs = getDocumentFrequencies(map(" ".join, tokensPerFile), self.subsetModel)

		arousedFiles = []
		nonArousedFiles = []
		for vectors, filename in zip(tokensPerFile, trainFilenames):
			if isAroused(filename):
				arousedFiles.append(" ".join(vectors))
			else:
				nonArousedFiles.append(" ".join(vectors))


		pairsAroused = itertools.filterfalse(lambda x: np.array_equal(*x), itertools.product(nonArousedFiles, nonArousedFiles))
		pairsNonAroused = itertools.filterfalse(lambda x: np.array_equal(*x), itertools.product(arousedFiles, arousedFiles))
		pairs = list(map(";".join, list(pairsAroused) + list(pairsNonAroused)))

		noPairs = itertools.filterfalse(lambda x: np.array_equal(*x), itertools.product(arousedFiles, nonArousedFiles))
		noPairs = list(map(";".join, noPairs))

		random.seed(42)
		random.shuffle(noPairs)
		random.shuffle(pairs)

		wln = WeightLearningNetwork(self.subsetModel, maxLen)
		self.weights = wln.run(pairs, noPairs, self.docFreqs)

	def train(self, trainFilenames):
		self.learnRepresentationWeights(trainFilenames)
		super().train(trainFilenames)

if __name__ == "__main__":
	#run e.g. as python -m classifiers.DeBoomClassifier --modelPath /storage/MA/GoogleNews-vectors-negative300.bin --plot data/Veroff/*

	import argparse, sys

	parser = argparse.ArgumentParser(description='Vectorize documents, sum the vectors, pass to SVM for classification')
	parser.add_argument("-v", help="Be more verbose (-vv for max verbosity)", action="count", default=0)
	parser.add_argument("--modelPath", help="Path to word2vec model", required=True)
	parser.add_argument("--plot", help="Plot the PCA of the document sum of files", nargs="+")
	parser.add_argument("--train", help="Path to document(s) to train from", nargs="+")
	parser.add_argument("--test", help="Path to document(s) to test against", nargs="+")

	args = parser.parse_args()

	if not args.plot and ( not args.test or not args.train ):
		parser.error("Supply either test and train or plot")

	logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[logging.WARN, logging.INFO, logging.DEBUG][min(args.v, 2)])

	try:
		classifier = DeBoomClassifier(args.modelPath)

		if args.train:
			classifier.train(args.train)

		if args.test:
			result = classifier.test(args.test)

		if args.plot:
			classifier.learnRepresentationWeights(args.plot)
			print(classifier.weights)
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