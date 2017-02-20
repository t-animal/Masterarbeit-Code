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

from pprint import pprint
from gensim.models import Word2Vec

class WikiIterator():

	def __init__(self, corpus):
		self.corpus = corpus

	def __iter__(self):
		self.texts = self.corpus.get_texts()
		return self

	def next(self):
		return self.texts.next()


def main(corpusPath, modelPath, overwrite=False):
	log.info("Started application")

	if modelPath is None:
		modelPath = datetime.now().strftime("data/model--%Y-%m-%d--%H:%M:%S")

	log.info("Loading corpus")
	corpus = g.corpora.wikicorpus.WikiCorpus(corpusPath)

	log.info("Training model")
	model = g.models.Word2Vec(min_count=1)

	newPath = ""
	while os.path.exists(modelPath) and not overwrite and not newPath == "overwrite":
		newPath = input("Path '{}' exists. Please enter a different path or 'overwrite': ".format(modelPath))
		if not newPath == "overwrite":
			modelPath = newPath

	log.info("Saving model")
	model.save(modelPath)



if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Train a word2vec model on a wikipedia corpus and save it to disc')
	parser.add_argument("-f", "--forceOverwrite", help="Force overwriting of existing model files", action="store_true")
	parser.add_argument("-v", help="Be more verbose", action="store_true")
	parser.add_argument("--modelPath", help="Where to save the word2vec model", default=None)
	parser.add_argument("corpusPath", help="Path to the wikipedia corpus to learn from")

	args = parser.parse_args()

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
					level=log.INFO if args.v else log.WARNING)

	try:
		main(args.corpusPath, args.modelPath, args.forceOverwrite)
	except KeyboardInterrupt:
		pass