#!../venv/bin/python

""" This file is a playground file for trying out how to train and store a
	word2vec model using gensim.

	see also: https://rare-technologies.com/word2vec-tutorial/
"""

import gensim as g
import os
import pdb
import re
import logging as log
import multiprocessing

from pprint import pprint
from gensim.models import Word2Vec

class WikiIterator():

	def __init__(self, corpus):
		self.corpus = corpus

	def __iter__(self):
		self.texts = self.corpus.get_texts()
		return self

	def __next__(self):
		return list(map(lambda x: x.decode("utf-8"), next(self.texts)))


def main(corpusPath, modelPath, dimensions, workers):
	log.info("Started application")

	if modelPath is None:
		modelPath = datetime.now().strftime("data/model--%Y-%m-%d--%H:%M:%S")

	log.info("Loading corpus")
	#provide an empty dictionary to suppress processing of the corpus to create it (we don't need it, only need the text). same for lemmatization
	corpus = g.corpora.wikicorpus.WikiCorpus(corpusPath, lemmatize = False, dictionary = {})

	testIterator = iter(WikiIterator(corpus))
	log.debug("Here are the first 100 words from the first two articles:")
	log.debug(" ".join(list(next(testIterator))[:100]))
	log.debug(" ".join(list(next(testIterator))[:100]))

	log.info("Training model")
	model = g.models.Word2Vec(WikiIterator(corpus), size = dimensions, workers = workers)

	log.info("Saving model")
	model.save(modelPath)
	model.wv.save_word2vec_format(modelPath + ".w2v")


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Train a word2vec model on a wikipedia corpus and save it to disc')
	parser.add_argument("-v", help="Be more verbose (repeat v for more verbosity)", action="count", default=0)
	parser.add_argument("--num",  "-n", help = "The maximum cpus to use (default: all). Leaves n cpus free if negative", type = int, default = multiprocessing.cpu_count())
	parser.add_argument("--force", help="Force overwriting of existing model files", action="store_true")
	parser.add_argument("--modelPath", help="Where to save the word2vec model", default=None)
	parser.add_argument("--dimensions", "-d", help="Dimensionality of the vectors to be learned", type=int, default=400)
	parser.add_argument("corpusPath", help="Path to the wikipedia corpus to learn from.")

	args = parser.parse_args()

	if args.num <= 0:
		args.num = multiprocessing.cpu_count() - max(1 - multiprocessing.cpu_count(), args.num)
	args.num = min(args.num, multiprocessing.cpu_count())

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
				level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	try:
		main(args.corpusPath, args.modelPath, args.dimensions, args.num)
	except KeyboardInterrupt:
		pass
