#!../venv/bin/python

import itertools
import logging as log
import numpy as np
import random
import scipy

from itertools import starmap
from numpy.linalg import norm

"""This is a playground file to test out rothe et al.'s paper "ultra dense word embeddings"
"""

def batcher(iterable, batchSize):
	batch = []
	for item in iterable:
		batch.append(item)

		if len(batch) == batchSize:
			yield batch
			batch = []

	if len(batch) > 0:
		yield batch

def seperateDifferentGroups(group1, group2, dimensions = 1):
	"""Equation (3), used to maximize distance of words in different groups. The words to which the
	   word vectors in the groups correspond must have opposing meaning with regard to the desired
	   information

	   :param group1: word vectors of the first group
	   :param group2: word vectors of the second group
	   :type group1: list of np.array
	   :type group2: list of np.array

	   :returns: int, the cost factor to optimize
	"""

	batchSize = 100
	learnRate = 5

	assert(group1 and group2)
	assert(group1[0].size == group2[0].size)

	d = group1[0].size
	dstar = dimensions
	P = np.matrix(np.eye(dstar, d))
	Q = np.matrix(scipy.stats.ortho_group.rvs(d, random_state = 42))

	def costFunction(ew, ev):
		"""The part of equation (3) under the sum, i.e. the cost function for one vector pair"""
		return norm(P * Q * (ew - ev).reshape(d, 1))

	def derivedCostFunction(ew, ev, row, col):
		"""The derivative of `costFunction` with respect to one entry in Q, specified by row and column.
		I.e.
		$d/dq_{row,col} ||PQ(e_w-e_v)|| = d/dq_{row,col} ||PQv|| = \frac{v_{row}(v_1q_{row,1} + v_2q_{row,2}...)}{||PQ(e_w-e_v)||}$
		All lines where row > dstar return 0 as these columns in P are all 0"""
		if row > dstar:
			return 0

		v = (ew-ev).reshape(d, 1)
		return v[col] * ((Q[row] * v)[(0,0)])/costFunction(ew, ev)

	def reorthogonalize(Q):
		U,S,V = np.linalg.svd(Q)
		newQ = U * V #linalg.svd returns V transposed
		assert((newQ * np.transpose(newQ) - np.identity(d) < 0.0001).all())
		return newQ


	random.seed(42)
	for iteration in itertools.count(1):
		L = list(itertools.product(group1, group2))
		random.shuffle(L)
		batches = batcher(L, batchSize)

		for batchNo, batch in enumerate(batches):
			cost = sum(starmap(costFunction, batch))/len(batch)
			log.info("Iteration #%3d, batch %4d: Cost is %9.5f, learnRate was %7.5f", iteration, batchNo, cost, learnRate)

			Qderiv = np.matrix(np.zeros((d,d)))
			for row, col in zip(range(dstar), range(d)):
				Qderiv[(row, col)] = sum([derivedCostFunction(ew, ev, row, col) for ew, ev in batch])/len(batch)

			Q -= learnRate * Qderiv
			Q = reorthogonalize(Q)

		#Q: does the learn rate decrease per iteration or per batch? paper reads like per iteration, even though that's weird
		learnRate *= 0.99


if __name__ == "__main__":

	import pickle
	from gensim.models import KeyedVectors


	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=log.DEBUG)

	try:
		with open("posVecs.pickle", "rb") as file:
			posVec = pickle.load(file)
		with open("negVecs.pickle", "rb") as file:
			negVec = pickle.load(file)
		log.info("Loaded vectors")
	except Exception as e:
		log.exception(e)
		model = KeyedVectors.load_word2vec_format("/storage/MA/customVectors/w2v-wiki/w2v.wikipedia-en.cbow.softmax.100d.w2v.gz")

		pos = "dedication achieves versatility fellowships recognized excellence refreshments versatility craftsmanship elegance guestrooms approachable celebrates partnering accompanist appreciates appoints toastmasters personable felicitated enthuses welcomes timeless chorale mentors expertise mentorship impressed musicality delighted unfailing honored learnings harmoniously appreciating appreciative volunteerism celebrate versatile unpretentious coachable strives telecompaper qualities recognizes richness intacct scottrade fluent vibrant celebrating dedication congratulate elegant supple talents handcrafted rotarians strived freshness showcasing fellowship honorees adaptability pedometers unrivaled enjoying teamwork honorees doctorate honored appreciation praises gracious volunteerism thank concierge secures stylish fulfills congratulations wonderful thoughtfulness invaluable harmonious commitment unsurpassed elects dedicated eventful visionary gratitude praising acumen concertmaster recognitions horsemanship appreciated accolade uniqueness complement spacious sennheiser affordable passion unparalleled exciting complementing commends congratulations inspiring exemplifies portico musicianship competencies strengths nurturing volunteering trailblazer "
		neg = "overreaction unsanitary vandalism pileup unhygienic underfunding bungle nooses landslides fester understaffing suffocation abusive stench gutless mixup faulty uncaring scaremongering abusive disgusting mismanagement inconsiderate sickening racist sickens cyberattacks looting firebombed racist libelous filthy gouging inhumane inaction misdiagnosed inaccurate flooding hallucinating kneeing unprofessional stink slur kidnappings xenophobic whiners blowback rioted clashes amok incoherent blamed feces fouls starvation loadshedding spat beatings stagflation unfunny scapegoating insult spygate inept reeked rioting asphyxiated overcrowding yobs uncooperative foul scuffles stabbing paranoid inaccuracy indefensible mishandling bungling fuming harassing irrational unprovoked urinating unpatriotic slurs squalid bungled uncontrollable threatening incompetent taunts tirade faulty vandalism untrustworthy derogatory irresponsible disgraceful urinated vindictiveness bloat disgusted blames festered mistreatment meddling irate malfunctioning harassed infighting fumes wrongheaded mismanaged angry brawl concussions slams lynchings nauseated underreporting desensitized shoddy rancid blaming sores misdiagnosis gangrene injures disorganization cowards unbalanced unfairness scapegoats looting suffocate undisciplined scumbag uninhabitable enraged fouling mistreat mistrial scuffle cramping"

		posVec = [model[w] for w in pos.split()]
		negVec = [model[w] for w in neg.split()]

		with open("posVecs.pickle", "wb") as file:
			pickle.dump(posVec, file)
		with open("negVecs.pickle", "wb") as file:
			pickle.dump(negVec, file)

	# posVec = [np.array([1,  2,  3,  4  ]).reshape(4,1), np.array([5  ,6,  7,  8, ]).reshape(4,1)]
	# negVec = [np.array([1.5,2.5,3.5,4.5]).reshape(4,1), np.array([5.5,6.5,7.5,8.5]).reshape(4,1)]
	log.info("Beginning seperation of %d positive and %d negative words", len(posVec), len(negVec))
	seperateDifferentGroups(posVec, negVec)