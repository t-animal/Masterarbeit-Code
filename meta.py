
from util import *
from util.argGenerator import *
from util.argGenerator import _dataSets
from itertools import chain, filterfalse
import gensim
import pickle

from gensim.models import KeyedVectors
from util.containers import LazyModel

def printData(set):
	set = [set]

	print("""
	* Samples:
		* Total: {}
			* {} Aroused
			* {} Non-Aroused
		* Test: {}
			*  {} Aroused
			*  {} Non-Aroused
		* Training:
			* Total: {}
				* {} Aroused
				* {} Non-Aroused
			* Train (80/20 split): {}
				* {} Aroused
				* {} Non-Aroused
			* Validate (80/20 split): {}
				*  {} Aroused
				*  {} Non-Aroused""".format(

	len(getAllFiles(set)),
	len(list(filter(isAroused, getAllFiles(set)))),
	len(list(filterfalse(isAroused, getAllFiles(set)))),

	len(list(generateTestset(set))),
	len(list(filter(isAroused, generateTestset(set)))),
	len(list(filterfalse(isAroused, generateTestset(set)))),

	len(generateTrainAndValidateset(set, set)[0] + generateTrainAndValidateset(set, set)[1]),
	len(list(filter(isAroused, generateTrainAndValidateset(set, set)[0] + generateTrainAndValidateset(set, set)[1]))),
	len(list(filterfalse(isAroused, generateTrainAndValidateset(set, set)[0] + generateTrainAndValidateset(set, set)[1]))),

	len(generateTrainAndValidateset(set, set)[0]),
	len(list(filter(isAroused, generateTrainAndValidateset(set, set)[0]))),
	len(list(filterfalse(isAroused, generateTrainAndValidateset(set, set)[0]))),

	len(generateTrainAndValidateset(set, set)[1]),
	len(list(filter(isAroused, generateTrainAndValidateset(set, set)[1]))),
	len(list(filterfalse(isAroused, generateTrainAndValidateset(set, set)[1]))),
	))


def getAvgLength(set):
	set = [set]

	fnames = getAllFiles(set)

	totalLength = 0
	totalLengthAroused = 0
	totalLengthNonAroused = 0

	for filename in fnames:
		with open(filename, "r") as file:
			totalLength += len(gensim.utils.simple_preprocess(file.read()))
	
	for filename in filter(isAroused, fnames):
		with open(filename, "r") as file:
			totalLengthAroused += len(gensim.utils.simple_preprocess(file.read()))
	
	for filename in filterfalse(isAroused, fnames):
		with open(filename, "r") as file:
			totalLengthNonAroused += len(gensim.utils.simple_preprocess(file.read()))

	return (totalLength / len(fnames), totalLengthAroused / len(list(filter(isAroused, fnames))), totalLengthNonAroused / len(list(filterfalse(isAroused, fnames))))



def getWords(modelPath, story):
	model = LazyModel(KeyedVectors.load_word2vec_format, modelPath, binary=modelPath.endswith("bin"))

	with open(story, "r") as file:
		content = file.read()
		storyTokens = list(gensim.utils.simple_preprocess(content))

	try:
		cachePath = os.path.split(story)
		cachePath = os.path.join(cachePath[0], "." + cachePath[1] + ".veccache")
		with open(cachePath, "rb") as cacheFile:
			cache = pickle.load(cacheFile, encoding='latin1')
			vectorizableTokens = [token for token, vector in cache[os.path.realpath(modelPath)]]

	except Exception:
		vectorizableTokens = [token for token in storyTokens if token in model]


	vecTokenSet = set(vectorizableTokens)
	missingWords = [token for token in storyTokens if token not in vecTokenSet]

	return storyTokens, vectorizableTokens, missingWords

def getMissingWords(modelPath, stories):
	words = [getWords(modelPath, story) for story in stories]
	_, __, missingWords = zip(*words)

	return set(chain.from_iterable(missingWords))