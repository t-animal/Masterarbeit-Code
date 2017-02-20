#!../venv/bin/python

""" This file is a playground file for trying out how to load pre-computed
    word vectors and convert a text document to a sequence of vectors

    see also: https://rare-technologies.com/word2vec-tutorial/
              http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
"""

import gensim as g
import pdb
import re
import logging as log

from pprint import pprint
from gensim.models import Word2Vec


def main(filenames, model="./data/GoogleNews-vectors-negative300.bin"):
	log.info("Started application")

	model = Word2Vec.load_word2vec_format(model, binary=True)

	log.info("Loaded model")

	for filename in filenames:
		with open(filename, "r") as file:
			log.info("Beginning with file %s", filename)

			content = file.read()
			content = re.sub('[,.-]', '', content)

			translatedFile = []

			for token in content.lower().split():
				if token not in model.wv.vocab.keys():
					log.info("token '%s' not in vocabulary", token)
					continue

				translatedFile.append((token, model[token]))

			pprint(translatedFile)

			log.info("Finished with file %s", filename)

	log.info("Finished")

if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Vectorize text document(s)')
	parser.add_argument("--model", help="Path to word2vec model")
	parser.add_argument("-v", help="Be more verbose", action="store_true")
	parser.add_argument("filenames", help="Path to document(s) to vectorize", nargs="+")

	args = parser.parse_args()

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=log.INFO if args.v else log.WARNING)

	try:
		if args.model:
			main(args.filenames, args.model)
		else:
			main(args.filenames)
	except KeyboardInterrupt:
		pass