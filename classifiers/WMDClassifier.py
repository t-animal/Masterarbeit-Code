import logging
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from util import isAroused
from util.containers import CachedKeyedVectors, TestresultContainer
from util.classifiers import EmbeddingsClassifier, SVMClassifierMixin

log = logging.getLogger("de.t_animal.MA.classifiers.WMDClassifier")

class WMDClassifier(EmbeddingsClassifier):

	def __init__(self, modelPath = None, k = 3):
		super().__init__(modelPath)
		self.distanceCalculator = CachedKeyedVectors(self.model)

		self.documents = []

		self.knn = KNeighborsClassifier(n_neighbors = k,
											algorithm = "brute",
											metric = self.wmdistanceMetric)

	def _generateDescribingVectors(self, filename):
		"""Vectorizes a file

		   :param filename: the file to process
		   :type filename: string
		   :returns: numpy array
		"""

		yield self.removeStopWords(self._vectorizeFile(filename))

	def wmdistanceMetric(self, x, y):
		document1 = list(self._getDescribingVectors(self.documents[int(x[0])])[0])
		document2 = list(self._getDescribingVectors(self.documents[int(y[0])])[0])

		distance = self.distanceCalculator.wmdistance(document1, document2)
		# distance = self.model.wmdistance([w for w,v in document1], [w for w,v in document2])
		# print("{}->{}: {}".format(os.path.basename(self.documents[int(x[0])]), os.path.basename(self.documents[int(y[0])]), distance))
		return distance

	def train(self, trainFilenames):
		"""Trains the classifier with the given filenames

		   :param trainFilenames: The filenames to train upon
		   :type trainFilenames: iterable of strings
		"""

		startIndex = len(self.documents)
		endIndex = startIndex + len(trainFilenames)
		self.documents += trainFilenames

		X = [[i] for i in range(startIndex, endIndex)]
		Y = [isAroused(f) for f in trainFilenames]

		self.knn.fit(np.array(X), np.array(Y))

	def test(self, testFilenames):

		startIndex = len(self.documents)
		endIndex = startIndex + len(testFilenames)
		self.documents += testFilenames

		X = [[i] for i in range(startIndex, endIndex)]

		log.info("Beginning testing")
		testResult = TestresultContainer(True, False, "aroused", "nonAroused")

		for x, filename in zip(X, testFilenames):
			result = self.knn.predict([x])[0]
			testResult.addResult(bool(result), isAroused(filename))

			log.info("Checked file %s, result %s (%s)", filename, result,
			         "CORRECT" if bool(result) == isAroused(filename) else "INCORRECT")

		log.info("Finished testing: %s", testResult.oneline())

		return testResult