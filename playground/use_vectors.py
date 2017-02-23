#!../venv/bin/python

""" This file is a playground file for trying out how to use the vectors obtained
	e.g. from vectorize and use them (here put them into a svm)

	see also: https://rare-technologies.com/word2vec-tutorial/
"""

import gensim as g
import pdb
import numpy as np
import os
import re
import logging as log
import traceback, sys, code
try:
	import cPickle as pickle
except ImportError:
	import pickle

from sklearn import svm as SVM
from pprint import pprint
from gensim.models import Word2Vec
from util.plot import plotPCA
from util.LazyModel import LazyModel

def getVectorSum(filename, model, modelPath=None):
	cachePath = os.path.split(filename)
	cachePath = os.path.join(cachePath[0], "." + cachePath[1] + ".veccache")
	try:
		with open(cachePath, "rb") as cacheFile:
			translatedFile = pickle.load(cacheFile, encoding='latin1')[modelPath]
			log.debug("loaded vectorized file %s from cache", filename)
	except Exception:
		translatedFile = []

		with open(filename, "r") as file:

			content = file.read()
			content = re.sub('[,.-]', '', content)

			for token in content.lower().split():
				if token not in model.wv.vocab.keys():
					log.debug("token '%s' not in vocabulary", token)
					continue

				translatedFile.append((token, model[token]))

		try:
			with open(cachePath, "wb") as cacheFile:
				log.debug("storing vectorized file %s to cache", filename)
				pickle.dump({modelPath: translatedFile}, cacheFile)
		except Exception as e:
			log.warn("Could not store cache: %s", str(e))

	fileSum = np.zeros(300, dtype=np.float64)
	for token, vector in translatedFile:
		fileSum += vector

	return fileSum

def main(testFilenames, trainFilenames=None, svmPath=None, modelPath=None):
	log.info("Started application")

	model = LazyModel(Word2Vec.load_word2vec_format, modelPath, binary=True)
	log.info("Loaded model")

	if trainFilenames:
		c_vectors = []
		e_vectors = []

		for filename in trainFilenames:
			log.info("Beginning with file %s", filename)
			fileSum = getVectorSum(filename, model)

			if "C" in filename:
				c_vectors.append(fileSum)
			else:
				e_vectors.append(fileSum)

			log.debug("sum_vector: %s", str(fileSum))
			log.info("Finished with file %s", filename)


		svm = SVM.SVC()
		log.info("Start training svm")
		svm.fit(c_vectors + e_vectors, ["c"] * len(c_vectors) + ["e"] * len(e_vectors))

		if svmPath:
			with open(svmPath, "wb") as persistanceFile:
				pickle.dump(svm, persistanceFile)
	elif svmPath:
		log.info("Loading SVM from pickled file")
		with open(svmPath, "rb") as persistanceFile:
			svm = pickle.load(persistanceFile, encoding='latin1')
	else:
		raise ValueError("Either svmPath or trainFilenames and model must be supplied")

	log.info("Beginning testing")
	correct = 0.
	c_vectors = []
	e_vectors = []
	for filename in testFilenames:
		fileSum = getVectorSum(filename, model)

		if "C" in filename:
			c_vectors.append(fileSum)
		else:
			e_vectors.append(fileSum)

		result = svm.predict([fileSum])[0]
		if result.upper() in filename:
			correct += 1

		log.info("Checked file %s, result %s", filename, result)

	print("{} of {} correct ({}%)".format(correct, len(testFilenames), correct/len(testFilenames)*100))

	for filename in trainFilenames:
		fileSum = getVectorSum(filename, model)

		if "C" in filename:
			c_vectors.append(fileSum)
		else:
			e_vectors.append(fileSum)

	plotPCA(c_vectors, e_vectors, "c", "e", 3)

	log.info("Finished")



if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Vectorize text document(s)')
	parser.add_argument("--modelPath", help="Path to word2vec model (defaults to ./data/GoogleNews-vectors-negative300.bin)", default="./data/GoogleNews-vectors-negative300.bin")
	parser.add_argument("--svmPath", help="Path to load the svm from if no trainFilenames are given or store it to if trainFilenames are given")
	parser.add_argument("-v", help="Be more verbose", action="store_true")
	parser.add_argument("--trainFilenames", help="Path to document(s) to train from", nargs="*")
	parser.add_argument("--testFilenames", help="Path to document(s) to test against", nargs="+")

	args = parser.parse_args()

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=log.INFO if args.v else log.WARNING)

	try:
		if args.modelPath:
			main(args.testFilenames, args.trainFilenames, args.svmPath, args.modelPath)
	except KeyboardInterrupt:
		pass
	except:
		type, value, tb = sys.exc_info()
		traceback.print_exc()
		last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
		frame = last_frame().tb_frame
		ns = dict(frame.f_globals)
		ns.update(frame.f_locals)
		code.interact(local=ns)
