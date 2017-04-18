from . import documentFrequency

import theano
import numpy as np

from threading import Condition

class AbstractProcessor(object):
	def __init__(self):
		self.lock = Condition()
		self.cont = False
		self.ready = False
		self.stop = False

	def new_epoch(self):
		self.begin_of_new_epoch()
		self.lock.acquire()
		self.cont = True
		self.ready = False
		self.stop = False
		self.lock.notifyAll()
		self.lock.release()

	def process(self):
		while True:
			self.lock.acquire()
			while not self.cont:
				self.lock.wait()
			self.ready = False
			self.cont = False
			self.lock.release()
			if self.stop:
				break

			self.process_batch()

			self.lock.acquire()
			self.ready = True
			self.cont = False
			self.lock.notifyAll()
			self.lock.release()

	def begin_of_new_epoch(self):
		"""Abstract"""
		raise NotImplementedError("Method begin_of_new_epoch is not implemented in this class.")

	def process_batch(self):
		"""Abstract"""
		raise NotImplementedError("Method process_all_batches is not implemented in this class.")


class PairProcessor(AbstractProcessor):
	def __init__(self, pairs, noPairs, docfreqs,
				 wordvectors, no_words=20, embedding_dim=400, batch_size=100):

		super(PairProcessor, self).__init__()

		if type(pairs) == str:
			with open(pairs) as pairsFile:
				self.pairs = [p for p in pairsFile]
		else:
			self.pairs = pairs

		if type(pairs) == str:
			with open(noPairs) as noPairsFile:
				self.noPairs = [p for p in noPairsFile]
		else:
			self.noPairs = noPairs

		self.batch_size = batch_size
		self.no_words = no_words
		self.embedding_dim = embedding_dim

		self.x1 = np.zeros((batch_size, embedding_dim, no_words), dtype=theano.config.floatX)
		self.x2 = np.zeros((batch_size, embedding_dim, no_words), dtype=theano.config.floatX)
		self.y = np.zeros((batch_size), dtype=theano.config.floatX)
		self.z = np.zeros((batch_size), dtype=theano.config.floatX)

		if type(docfreqs) == str:
			documentFrequency.load(docfreqs)
		else:
			self.docfreqs = docfreqs

		if type(wordvectors) == str:
			self.wordvectors = gensim.models.Word2Vec.load_word2vec_format(wordvectors, binary = wordvectors.endswith(".bin"))
		else:
			self.wordvectors = wordvectors

	def begin_of_new_epoch(self):
		self.pairIterator = iter(self.pairs)
		self.noPairIterator = iter(self.noPairs)

	def process_batch(self):
		for i in range(0, self.batch_size, 2):
			pair = next(self.pairIterator).split(';')
			no_pair = next(self.noPairIterator).split(';')

			pairA = pair[0].split()
			pairB = pair[1].split()
			no_pairA = no_pair[0].split()
			no_pairB = no_pair[1].split()

			dA = [0]*self.no_words
			dB = [0]*self.no_words
			nA = [0]*self.no_words
			nB = [0]*self.no_words

			for k in range(self.no_words):
				dA[k] = self.docfreqs[self.wordvectors.vocab[pairA[k]].index]
				dB[k] = self.docfreqs[self.wordvectors.vocab[pairB[k]].index]
				nA[k] = self.docfreqs[self.wordvectors.vocab[no_pairA[k]].index]
				nB[k] = self.docfreqs[self.wordvectors.vocab[no_pairB[k]].index]

			_, pairA = zip(*sorted(zip(dA, pairA)))
			_, pairB = zip(*sorted(zip(dB, pairB)))
			_, no_pairA = zip(*sorted(zip(nA, no_pairA)))
			_, no_pairB = zip(*sorted(zip(nB, no_pairB)))

			for j in range(self.no_words):
				self.x1[i, :, j] = self.wordvectors[pairA[j]]
				self.x2[i, :, j] = self.wordvectors[pairB[j]]
				self.x1[i+1, :, j] = self.wordvectors[no_pairA[j]]
				self.x2[i+1, :, j] = self.wordvectors[no_pairB[j]]

			self.y[i] = 0.0
			self.z[i] = -1.0
			self.y[i+1] = 1.0
			self.z[i+1] = 1.0


class LengthTweetPairProcessor(PairProcessor):
	def __init__(self, pairs_filename='pairs.txt', no_pairs_filename='no_pairs.txt', docfreq_filename='docfreqs.npy',
				 wordvectors='minimal', no_words=30, embedding_dim=400, batch_size=100, cutoff_function=None):
		## no_words is the maximum number of words allowed
		super(LengthTweetPairProcessor, self).__init__(pairs_filename, no_pairs_filename,
											   docfreq_filename, wordvectors, no_words,
											   embedding_dim, batch_size)

		self.l1 = np.zeros((batch_size), dtype=theano.config.floatX)
		self.l2 = np.zeros((batch_size), dtype=theano.config.floatX)
		self.cutoff_function = cutoff_function
		self.indices1 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)
		self.indices2 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)

	def process_batch(self):
		self.indices1[:, :] = 0.0
		self.indices2[:, :] = 0.0
		for i in range(0, self.batch_size, 2):
			pair = next(self.pairIterator).split(';')
			no_pair = next(self.noPairIterator).split(';')

			pairA = [k for k in pair[0].split() if k in self.wordvectors]
			pairB = [k for k in pair[1].split() if k in self.wordvectors]
			no_pairA = [k for k in no_pair[0].split() if k in self.wordvectors]
			no_pairB = [k for k in no_pair[1].split() if k in self.wordvectors]

			dA = [0]*len(pairA)
			dB = [0]*len(pairB)
			nA = [0]*len(no_pairA)
			nB = [0]*len(no_pairB)

			for k in range(len(pairA)):
				dA[k] = self.docfreqs[self.wordvectors.vocab[pairA[k]].index]
			for k in range(len(pairB)):
				dB[k] = self.docfreqs[self.wordvectors.vocab[pairB[k]].index]
			for k in range(len(no_pairA)):
				nA[k] = self.docfreqs[self.wordvectors.vocab[no_pairA[k]].index]
			for k in range(len(no_pairB)):
				nB[k] = self.docfreqs[self.wordvectors.vocab[no_pairB[k]].index]

			_, pairA = zip(*sorted(zip(dA, pairA)))
			_, pairB = zip(*sorted(zip(dB, pairB)))
			_, no_pairA = zip(*sorted(zip(nA, no_pairA)))
			_, no_pairB = zip(*sorted(zip(nB, no_pairB)))

			self.x1[i, :, :] = 0.0
			self.x2[i, :, :] = 0.0
			self.x1[i+1, :, :] = 0.0
			self.x2[i+1, :, :] = 0.0
			for j in range(len(pairA)):
				self.x1[i, :, j] = self.wordvectors[pairA[j]]
			for j in range(len(pairB)):
				self.x2[i, :, j] = self.wordvectors[pairB[j]]
			for j in range(len(no_pairA)):
				self.x1[i+1, :, j] = self.wordvectors[no_pairA[j]]
			for j in range(len(no_pairB)):
				self.x2[i+1, :, j] = self.wordvectors[no_pairB[j]]

			self.l1[i] = len(pairA) - 1
			self.l2[i] = len(pairB) - 1
			self.l1[i+1] = len(no_pairA) - 1
			self.l2[i+1] = len(no_pairB) - 1
			self.y[i] = 0.0
			self.z[i] = -1.0
			self.y[i+1] = 1.0
			self.z[i+1] = 1.0

			assert all([self.l1[i] == int(self.l1[i]), self.l1[i+1] == int(self.l1[i+1]), self.l2[i+1] == int(self.l2[i+1]), self.l2[i+1] == int(self.l2[i+1])])
			# import pdb
			# pdb.set_trace()

			self.indices1[0:int(self.l1[i])+1, i] = np.transpose(np.linspace(0,
													(1 - self.cutoff_function(self.l1[i] + 1))*(self.no_words - 1), self.l1[i] + 1))
			self.indices2[0:int(self.l2[i])+1, i] = np.transpose(np.linspace(0,
													(1 - self.cutoff_function(self.l2[i] + 1))*(self.no_words - 1), self.l2[i] + 1))
			self.indices1[0:int(self.l1[i+1])+1, i+1] = np.transpose(np.linspace(0,
													(1 - self.cutoff_function(self.l1[i+1] + 1))*(self.no_words - 1), self.l1[i+1] + 1))
			self.indices2[0:int(self.l2[i+1])+1, i+1] = np.transpose(np.linspace(0,
													(1 - self.cutoff_function(self.l2[i+1] + 1))*(self.no_words - 1), self.l2[i+1] + 1))