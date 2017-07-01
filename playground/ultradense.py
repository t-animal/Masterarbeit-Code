#!../venv/bin/python

import datetime
import itertools
import logging as log
import numpy as np
import math
import random
import scipy

from datetime import datetime
from itertools import starmap
from numpy.linalg import norm

import pickle
from gensim.models import KeyedVectors

"""This is a playground file to test out rothe et al.'s paper "ultra dense word embeddings"
"""

def batches(iterable, batchSize = 200):
	batch = []
	for item in iterable:
		batch.append(item)

		if len(batch) == batchSize:
			yield batch
			batch = []

	if len(batch) > 0:
		yield batch

def seperateDifferentGroups(group1, group2, dimensions = 1, batchSize = 300, alpha = 0.8):
	"""Equation (3), used to minimize (and maximize) distance of groups of words of (opposing) meaning in a subspace.
	   The words to which the word vectors in the groups correspond must have opposing meaning with regard to the desired
	   information

	   :param group1: word vectors of the first group (e.g. positive sentiment)
	   :param group2: word vectors of the second group (e.g. negative sentiment)
	   :param dimensions: the number of dimensions to fit the information to
	   :param batchSize: the batchSize to use
	   :param alpha: combination factor alpha (how much influence [0 <= alpha <= 1]
	   :type group1: list of np.array
	   :type group2: list of np.array
	   :type dimensions: int
	   :type batchSize: int
	   :type alpha: float

	   :returns: np.matrix, the optimized transformation matrix
	"""

	#we take our learning rate from the plot in their repo, showing that the sweet spot lies at about the 400th iteration
	#when starting from a learning rate of 1 => 1*0.99^400 = 0.017 and ends at the 430th iteration 1*0.99^430 => 0.013
	#and add some space around it
	learnRate = 5 * 0.22 #paper starts at 5, but that's very inefficient, and this implementation is horribly inefficient, so prune that
	abortRate = 5 * 0.008 #at which learning rate to abort

	assert(group1 and group2)
	assert(group1[0].size == group2[0].size)

	d = group1[0].size
	dstar = dimensions
	P = np.matrix(np.eye(dstar, d))
	Q = np.matrix(scipy.stats.ortho_group.rvs(d, random_state = 42))

	costCache = {}
	def costFunction(ew, ev):
		"""The part of equation (3) and (4) under the sum, i.e. the cost function for one vector pair"""
		key = (id(ew), id(ev))
		if key in costCache:
			return costCache[key]

		costCache[key] = norm(P * Q * (ew - ev).reshape(d, 1))
		return costCache[key]

	def derivedCostFunction(ew, ev, row, col):
		"""The derivative of `costFunction` with respect to one entry in Q, specified by row and column.
		I.e.
		$d/dq_{row,col} ||PQ(e_w-e_v)|| = d/dq_{row,col} ||PQv|| = \frac{v_{row}(v_1q_{row,1} + v_2q_{row,2}...)}{||PQ(e_w-e_v)||}$
		All lines where row >= dstar return 0 as the corresponding columns in P are all 0"""
		if row >= dstar:
			return 0

		assert(not all(ew == ev))

		v = (ew-ev).reshape(d, 1)
		return v[col] * ((Q[row] * v)[(0,0)])/costFunction(ew, ev)

	def reorthogonalize(Q):
		U,S,V = np.linalg.svd(Q)
		newQ = U * V #linalg.svd returns V transposed
		assert((newQ * np.transpose(newQ) - np.identity(d) < 0.00001).all())
		return newQ


	random.seed(42)
	startTime = datetime.now()
	startCost = sum(starmap(costFunction, itertools.product(group1, group2)))
	for iteration in itertools.count(1):
		LDiffGroup = list(itertools.product(group1, group2))
		LSameGroup = list(itertools.combinations(group1, 2)) + list(itertools.combinations(group2, 2))
		random.shuffle(LDiffGroup)
		random.shuffle(LSameGroup)

		for ew, ev in LSameGroup:
			assert(not all(ew == ev))
		for ew, ev in LDiffGroup:
			assert(not all(ew == ev))

		for batchNo, (batchDiffGroup, batchSameGroup) in enumerate(zip(batches(LDiffGroup), batches(LSameGroup))):
			if batchNo%5 == 0:
				costSameGroup = sum(starmap(costFunction, batchSameGroup))/len(batchSameGroup)
				costDiffGroup = -sum(starmap(costFunction, batchDiffGroup))/len(batchDiffGroup)
				cost = ((1-alpha) *  costDiffGroup + alpha * costSameGroup)
				log.info("Iteration #%3d, batch %4d: Cost is %9.5f (same: %9.5f, diff: %9.5f), learnRate was %8.6f",
				         iteration, batchNo, cost, costSameGroup, costDiffGroup, learnRate)

			QDerivedDiffGroup = np.matrix(np.zeros((d,d)))
			for row in range(dstar):
				for col in range(d):
					QDerivedDiffGroup[(row, col)] = -sum([derivedCostFunction(ew, ev, row, col) for ew, ev in batchDiffGroup])/len(batchDiffGroup)

			QDerivedSameGroup = np.matrix(np.zeros((d,d)))
			for row in range(dstar):
				for col in range(d):
					QDerivedSameGroup[(row, col)] = sum([derivedCostFunction(ew, ev, row, col) for ew, ev in batchSameGroup])/len(batchSameGroup)

			Q -= learnRate * (alpha *  QDerivedDiffGroup + (1 - alpha) * QDerivedSameGroup)
			Q = reorthogonalize(Q)

			#invalidate costCache
			costCache = {}

			#does the learn rate decrease per iteration or per batch? paper reads like per iteration, but that's weird
			#judging from their code they draw exactly one random batch per iteration, i.e. it's per iteration
			learnRate *= 0.99

			if learnRate < abortRate:
				break

		if learnRate < abortRate:
			break

	endTime = datetime.now()
	totalCost = sum(starmap(costFunction, itertools.product(group1, group2)))
	log.info("Terminating at after %d iterations over %d batches with termanation learnRate %f.", iteration, batchNo, learnRate)
	log.info("Training took %s", endTime - startTime)
	log.info("Cost with random Q was %f (averaged over all pairs: %f)", startCost, startCost/len(group1)/len(group2))
	log.info("Final total cost is %f (averaged over all pairs: %f)", totalCost, totalCost/len(group1)/len(group2))

	with open("final_Q-{}.pickle".format(datetime.now()), "wb") as file:
		pickle.dump({
			"parameters": {
				"alpha": alpha,
				"batchSize": batchSize},
			"iterations": iteration,
			"batches": batchNo,
			"learnRate": learnRate,
			"totalCost": totalCost,
			"avgCost": totalCost/len(group1)/len(group2),
			"Q": Q
			}, file)

	return Q


def checkResults(posVec, negVec, Q = None):
	if Q is None:
		Q = np.identity(posVec[0].size)

	pospos=0
	posneg=0
	negpos=0
	negneg=0
	poszero = 0
	negzero = 0
	for vec in posVec:
		if (Q*vec.reshape(vec.size,1))[(0,0)] > 0:
			pospos += 1
		elif (Q*vec.reshape(vec.size,1))[(0,0)] < 0:
			posneg += 1
		else:
			import pdb
			pdb.set_trace()
			poszero += 1
	for vec in negVec:
		if (Q*vec.reshape(vec.size,1))[(0,0)] > 0:
			negpos += 1
		elif (Q*vec.reshape(vec.size,1))[(0,0)] < 0:
			negneg += 1
		else:
			negzero +=1

	print("Positive Words > 0: {}".format(pospos))
	print("Positive Words < 0: {}".format(posneg))
	print("Positive Words = 0: {}".format(poszero))

	print("Negative Words > 0: {}".format(negpos))
	print("Negative Words < 0: {}".format(negneg))
	print("Negative Words = 0: {}".format(negzero))



if __name__ == "__main__":

	import pickle
	from gensim.models import KeyedVectors


	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=log.DEBUG)

	try:
		with open("posVecs.pickle", "rb") as file:
			posVec, posTestVec = pickle.load(file)
		with open("negVecs.pickle", "rb") as file:
			negVec, negTestVec = pickle.load(file)
		log.info("Loaded vectors")
	except Exception as e:
		log.exception(e)
		model = KeyedVectors.load_word2vec_format("/storage/MA/GoogleNews-vectors-negative300.bin", binary=True)
		frequentWords = model.index2word[:80000]

		posWords = [line[:-1] for line in open("/storage/MA/sentimentLexica/positive-words.txt")
		                      if not line.startswith(";") and len(line) > 2 and line[:-1] in frequentWords]
		negWords = [line[:-1] for line in open("/storage/MA/sentimentLexica/negative-words.txt")
		                      if not line.startswith(";") and len(line) > 2 and line[:-1] in frequentWords]

		random.seed(42)
		random.shuffle(posWords)
		random.shuffle(negWords)

		posVec = [model[w] for w in posWords[:200]]
		negVec = [model[w] for w in negWords[:200]]

		posTestVec = [model[w] for w in posWords[201:401]]
		negTestVec = [model[w] for w in negWords[201:401]]

		with open("posVecs.pickle", "wb") as file:
			pickle.dump((posVec, posTestVec), file)
		with open("negVecs.pickle", "wb") as file:
			pickle.dump((negVec, negTestVec), file)

	import sys
	alpha = float(sys.argv[1])
	print("Alpha set to " + str(alpha))
	log.info("Beginning seperation of %d positive and %d negative words", len(posVec), len(negVec))
	Q = seperateDifferentGroups(posVec, negVec, alpha=alpha)

	print("Alpha set to " + str(alpha))
	log.info("Checking results. This is the baseline: ")
	checkResults(posTestVec, negTestVec)
	log.info("Checking results. This is the result using the trained Q: ")
	checkResults(posTestVec, negTestVec, Q)

