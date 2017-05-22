#!../venv/bin/python

""" This file is a playground file for trying out how to train and store a
	word2vec model using gensim.

	see also: https://rare-technologies.com/word2vec-tutorial/
"""

import io
import gensim as g
import os
import pdb
import re
import sys
import logging as log
import multiprocessing
import subprocess
import threading

from pprint import pprint
from gensim.models import Word2Vec

from util.containers import FanFictionContainer, MultiGenreFanFictionContainer

class IterableWikiTexts():

	def __init__(self, corpus):
		self.corpus = corpus

	def __iter__(self):
		for text in self.corpus.get_texts():
			yield list(map(lambda x: x.decode("utf-8"), text))


class LogAdapter(threading.Thread):

	def __init__(self, logname, level = log.INFO):
		super().__init__()
		self.log = log.getLogger(logname)
		self.readpipe, self.writepipe = os.pipe()

		self.setLevel(level)

	def fileno(self):
		#when fileno is called this indicates the subprocess is about to fork => start thread
		self.start()
		return self.writepipe

	def run(self):
		inputFile = os.fdopen(self.readpipe)

		while True:
			line = inputFile.readline()

			if len(line) == 0:
				#no new data was added
				break

			self.logFunction(line.strip())

		self.log.warning("LogAdapter is done.")

	def setLevel(self, level):
		logFunctions = {
			log.DEBUG: self.log.debug,
			log.INFO: self.log.info,
			log.WARN: self.log.warn,
			log.ERROR: self.log.warn,
		}

		try:
			self.logFunction = logFunctions[level]
		except KeyError:
			self.logFunction = self.log.info

	def finished(self):
		"""If the filedescriptor is not closed this thread will prevent exiting. You can use this method
		   to clean up after the subprocess has terminated. """
		os.close(self.writepipe)

	def __enter__(self):
		return self

	def __exit__(self,  type, value, traceback):
		self.finished()


def word2vecWiki(corpusPath, modelPath, dimensions, workers):
	log.info("Started training")

	if modelPath is None:
		modelPath = datetime.now().strftime("w2v--wiki--%Y-%m-%d--%H:%M:%S")

	log.info("Loading corpus")
	#provide an empty dictionary to suppress processing of the corpus to create it (we don't need it, only need the text). same for lemmatization
	corpus = g.corpora.wikicorpus.WikiCorpus(corpusPath, lemmatize = False, dictionary = {})

	_word2vec(IterableWikiTexts(corpus), modelPath, dimensions, workers)


def word2vecFanFic(corpusPaths, modelPath, dimensions, workers):
	log.info("Started training")

	if modelPath is None:
		modelPath = datetime.now().strftime("w2v--fanfic--%Y-%m-%d--%H:%M:%S")

	log.info("Loading corpus")
	corpus = MultiGenreFanFictionContainer(*[FanFictionContainer(path) for path in corpusPaths])

	_word2vec(corpus, modelPath, dimensions, workers)


def _word2vec(corpus, modelPath, dimensions, workers):
	testIterator = iter(corpus)

	log.debug("Here are the first 10 words from the first two texts:")
	log.debug(" ".join(list(next(testIterator))[:10]))
	log.debug(" ".join(list(next(testIterator))[:10]))

	log.info("Training model")
	model = g.models.Word2Vec(corpus, size = dimensions, workers = workers)

	log.info("Saving model")
	model.save(modelPath)
	model.wv.save_word2vec_format(modelPath + ".w2v")



def gloveWiki(corpusPath, modelPath, dimensions, workers, memory = 4, maxVocab = 1e12, minCount = 10, skipSteps = 0, tempDir = "."):
	log.info("Started training")

	if modelPath is None:
		modelPath = datetime.now().strftime("glove--wiki--%Y-%m-%d--%H:%M:%S")

	log.info("Loading corpus")
	#provide an empty dictionary to suppress processing of the corpus to create it (we don't need it, only need the text). same for lemmatization
	corpus = g.corpora.wikicorpus.WikiCorpus(corpusPath, lemmatize = False, dictionary = {})

	_glove(IterableWikiTexts(corpus),  modelPath, dimensions, workers, memory, maxVocab, minCount, skipSteps, tempDir)


def gloveFanFic(corpusPaths, modelPath, dimensions, workers, memory = 4, maxVocab = 1e12, minCount = 10, skipSteps = 0, tempDir = "."):
	log.info("Started training")

	if modelPath is None:
		modelPath = datetime.now().strftime("glove--fanfic--%Y-%m-%d--%H:%M:%S")

	log.info("Loading corpus")
	corpus = MultiGenreFanFictionContainer([FanFictionContainer(path) for path in corpusPaths])

	_glove(corpus,  modelPath, dimensions, workers, memory, maxVocab, minCount, skipSteps, tempDir)


def _glove(corpus, modelPath, dimensions, workers, memory = 4, maxVocab = 1e12, minCount = 10, skipSteps = 0, tempDir = "."):
	testIterator = iter(corpus)
	log.debug("Here are the first 100 words from the first two texts:")
	log.debug(" ".join(list(next(testIterator))[:100]))
	log.debug(" ".join(list(next(testIterator))[:100]))

	#Call vocab_count and pass the wikipedia content on stdin
	if skipSteps < 1:
		with LogAdapter("de.t_animal.MA.glovebinaries.vocab_count", level=log.DEBUG) as stderr2log:
			with open(tempDir + "/vocabcount.cache", "w") as outFile:
				with subprocess.Popen("../util/glove/vocab_count -max-vocab {:.0f} "
					                  "-min-count {} -verbose 2".format(maxVocab, minCount).split(" "),
				                      stdin = subprocess.PIPE, stdout = outFile, stderr = stderr2log, bufsize = 0) as counter:

					for index, line in enumerate(corpus):
						if index > 0 and index % 10000 == 0:
							log.info("Processing article no {} for vocab counting".format(index))
						counter.stdin.write((" ".join(line) + " ").encode("utf-8"))

			log.info("Iterated over all articles for vocab counting")
			counter.wait()
			if not counter.returncode == 0:
				log.error("Vocabulary counting failed with exit code {}".format(counter.returncode))

		log.info("Vocabulary counting finished. Beginning coocurrence calculation")
	else:
		log.info("Vocabulary counting skipped. Beginning coocurrence calculation")

	#Call cooccur and pass the wikipedia content on stdin
	if skipSteps < 2:
		with LogAdapter("de.t_animal.MA.glovebinaries.cooccur", level=log.DEBUG) as stderr2log:
			with open(tempDir + "/coocurrence.cache", "w") as outFile:
				with subprocess.Popen("../util/glove/cooccur -memory {} -overflow-file {}/overflow "
					                  "-vocab-file {}/vocabcount.cache -verbose 2".format(memory, tempDir, tempDir).split(" "),
				                 stdin = subprocess.PIPE, stdout = outFile, stderr = stderr2log, bufsize = 0) as cooccurer:

					for index, line in enumerate(corpus):
						if index > 0 and index % 10000 == 0:
							log.info("Processing article no {} for coocurrence calculation".format(index))
						cooccurer.stdin.write((" ".join(line) + " ").encode("utf-8"))

			log.info("Iterated over all articles for coocurrence calculation. Continuing with output from cooccur")
			stderr2log.setLevel(log.DEBUG)
			cooccurer.wait()
			if not cooccurer.returncode == 0:
				log.error("Coocurrence counting failed with exit code {}".format(counter.returncode))

		log.info("Coocurrence calculation finished. Beginning shuffling.")
		log.info("These were the steps that are hard to do outside of this script. The rest can be done in the shell, too.")
	else:
		log.info("Coocurrence calculation skipped. Beginning shuffling.")

	if skipSteps < 3:
		#Call shuffle on the files created by the 2 previous calls
		with LogAdapter("de.t_animal.MA.glovebinaries.shuffle", level=log.INFO) as stderr2log:
			with open(tempDir + "/coocurrence-shuffled.cache", "wb") as outFile:
				with open(tempDir + "/coocurrence.cache", "rb") as inFile:
					shuffler = subprocess.Popen("../util/glove/shuffle -memory {} -temp-file {}/temp_shuffle -verbose 2".format(memory, tempDir).split(" "),
					                            stdin = inFile, stdout = outFile, stderr = stderr2log, bufsize = 0)
					shuffler.wait()

			if not shuffler.returncode == 0:
				log.error("Shuffling failed with exit code {}".format(counter.returncode))

		log.info("Shuffling finished. Beginning vector calculation.")
	else:
		log.info("Shuffling skipped. Beginning vector calculation.")

	#Call glove on the files created by the previous calls
	with LogAdapter("de.t_animal.MA.glovebinaries.glove", level=log.INFO) as stderr2log:
		glove = subprocess.Popen("../util/glove/glove -vector-size {size} -save-file {path} -threads {threads} "
		                 "-input-file {tmp}/coocurrence-shuffled.cache -vocab-file {tmp}/vocabcount.cache -verbose 2 -binary 0"
		                  .format(size=dimensions, path=modelPath, threads=workers, tmp=tempDir).split(" "), stderr=stderr2log)
		glove.wait()



if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Train a glove/word2vec model on a wikipedia corpus and save it to disc')
	parser.add_argument("-v", help="Be more verbose (repeat v for more verbosity)", action="count", default=0)
	parser.add_argument("--num",  "-n", help = "The maximum cpus to use (default: all). Leaves n cpus free if negative", type = int, default = multiprocessing.cpu_count())
	parser.add_argument("--modelPath", help="Where to save the word2vec model", default=None)
	parser.add_argument("--dimensions", "-d", help="Dimensionality of the vectors to be learned", type=int, default=400)
	parser.add_argument("--fanfiction", help="train on a fanfiction corpus, not on wikipedia", action="store_true")
	parser.add_argument("corpusPaths", help="Path(s) to the corpus file to learn from (one for the wikipedia dump, one or more for the fanfiction corpus).", nargs="+")

	modelParsers = parser.add_subparsers(title="action", help="Which model to train", dest="model")

	w2vparser = modelParsers.add_parser("w2v", help="Train a word2vec model")

	gloveParser = modelParsers.add_parser("glove", help="Train a glove model")
	gloveParser.add_argument("--memory",  help = "Soft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate", type = float, default = 4.0)
	gloveParser.add_argument("--maxVocab", help="Upper bound on vocabulary size, i.e. keep the <int> most frequent words. The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet.", type=int, default=1e12)
	gloveParser.add_argument("--minCount", help="Lower limit such that words which occur fewer than <int> times are discarded", type=int, default=10)
	gloveParser.add_argument("--skipSteps", help="Lets you skip the first N steps, e.g. if the produced files are still available", type=int, choices=[0,1,2,3], default=0)
	gloveParser.add_argument("--tempDir", help="Where to put (potentially large) temporary files", default=".")

	args = parser.parse_args()

	if args.num <= 0:
		args.num = multiprocessing.cpu_count() - max(1 - multiprocessing.cpu_count(), args.num)
	args.num = min(args.num, multiprocessing.cpu_count())

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
				level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	try:
		if args.fanfiction:
			if args.model == "w2v":
				word2vecFanFic(args.corpusPaths, args.modelPath, args.dimensions, args.num)
			else:
				gloveFanFic(args.corpusPaths, args.modelPath, args.dimensions, args.num, args.memory, args.maxVocab, args.minCount, args.skipSteps, args.tempDir)
		else:
			if args.model == "w2v":
				word2vecWiki(args.corpusPaths[0], args.modelPath, args.dimensions, args.num)
			else:
				gloveWiki(args.corpusPaths[0], args.modelPath, args.dimensions, args.num, args.memory, args.maxVocab, args.minCount, args.skipSteps, args.tempDir)

	except KeyboardInterrupt:
		pass
