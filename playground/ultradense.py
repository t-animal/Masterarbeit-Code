#!../venv/bin/python

import itertools
import logging as log
import numpy as np
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
	alpha = 5

	assert(group1 and group2)
	assert(group1[0].size == group2[0].size)

	d = group1[0].size
	dstar = dimensions
	P = np.eye(dstar, d)
	Q = scipy.stats.ortho_group.rvs(d, random_state = 42)

	def costFunction(ew, ev):
		return norm(P * Q * (ew - ev))

	def reorthogonalize(Q):
		U,S,V = numpy.linalg.svd(Q)
		return U * np.transpose(V)

	for iteration in itertools.count(1):
		L = itertools.product(group1, group2)
		batches = batcher(L, batchSize)

		for batchNo, batch in enumerate(batches):
			cost = sum(starmap(costFunction, L))
			log.info("Iteration #%d, batch %d: Cost is %d, alpha was %d", iteration, batchNo, cost, alpha)
			import pdb
			pdb.set_trace()

			grad = np.gradient(cost)
			Q = reorthogonalize(Q + alpha * grad)
			alpha *= 0.99


if __name__ == "__main__":

	import pickle
	from gensim.models import KeyedVectors


	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=log.INFO)

	try:
		with open("posVecs.pickle", "rb") as file:
			posVec = pickle.load(file)
		with open("negVecs.pickle", "rb") as file:
			negVec = pickle.load(file)
	except Exception as e:
		log.exception(e)
		model = KeyedVectors.load_word2vec_format("/storage/MA/customVectors/w2v-wiki/w2v.wikipedia-en.cbow.softmax.400d.w2v.gz")

		pos = "dedication achieves versatility fellowships recognized excellence refreshments versatility craftsmanship elegance guestrooms approachable celebrates partnering accompanist appreciates appoints toastmasters personable felicitated enthuses welcomes timeless chorale mentors expertise mentorship impressed musicality delighted unfailing honored learnings harmoniously appreciating appreciative volunteerism celebrate versatile unpretentious coachable strives telecompaper qualities recognizes richness intacct scottrade fluent vibrant celebrating dedication congratulate elegant supple talents handcrafted rotarians strived freshness showcasing fellowship honorees adaptability pedometers unrivaled enjoying teamwork honorees doctorate honored appreciation praises gracious volunteerism thank concierge secures stylish fulfills congratulations wonderful thoughtfulness invaluable harmonious commitment unsurpassed elects dedicated eventful visionary gratitude praising acumen concertmaster recognitions horsemanship appreciated accolade uniqueness complement spacious sennheiser affordable passion unparalleled exciting complementing commends congratulations inspiring exemplifies portico musicianship competencies strengths nurturing volunteering trailblazer "
		neg = "overreaction unsanitary vandalism pileup unhygienic underfunding bungle nooses landslides fester understaffing suffocation abusive stench gutless mixup faulty uncaring scaremongering abusive disgusting mismanagement inconsiderate sickening racist sickens cyberattacks looting firebombed racist libelous filthy gouging inhumane inaction misdiagnosed inaccurate flooding hallucinating kneeing unprofessional stink slur kidnappings xenophobic whiners blowback rioted clashes amok incoherent blamed feces fouls starvation loadshedding spat beatings stagflation unfunny scapegoating insult spygate inept reeked rioting asphyxiated overcrowding yobs uncooperative foul scuffles stabbing paranoid inaccuracy indefensible mishandling bungling fuming harassing irrational unprovoked urinating unpatriotic slurs squalid bungled uncontrollable threatening incompetent taunts tirade faulty vandalism untrustworthy derogatory irresponsible disgraceful urinated vindictiveness bloat disgusted blames festered mistreatment meddling irate malfunctioning harassed infighting fumes wrongheaded mismanaged angry brawl concussions slams lynchings nauseated underreporting desensitized shoddy rancid blaming sores misdiagnosis gangrene injures disorganization cowards unbalanced unfairness scapegoats looting suffocate undisciplined scumbag uninhabitable enraged fouling mistreat mistrial scuffle cramping"

		posVec = [model[w] for w in pos.split()]
		negVec = [model[w] for w in neg.split()]

		with open("posVecs.pickle", "wb") as file:
			pickle.dump(posVec, file)
		with open("negVecs.pickle", "wb") as file:
			pickle.dump(negVec, file)

	seperateDifferentGroups(posVec, negVec)