import gzip
import io
import json
import logging
import numpy
import scipy
import scipy.stats
import tarfile
import textwrap

from collections import OrderedDict
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab
from gensim.utils import simple_preprocess

class MultiGenreFanFictionContainer():
	"""Allows iteration over multiple FanFictionContainers, taking one item from each container after the other.
	   Balances genres by yielding the same number of elements from each container"""

	def __init__(self, ffcontainers, maxPerGenre = 1500000):
		self.containers = ffcontainers
		self.maxPerGenre = maxPerGenre

		containerLengths  = [len(c) for c in ffcontainers]
		if any(c < maxPerGenre for c in containerLengths):
			self.maxPerGenre = min(containerLengths)
			log.warning("At least one container is short than maxPerGenre "
				        "(%d) elements. Truncating each container to %d", maxPerGenre, self.maxPerGenre)

	def __iter__(self):
		iterators = list(map(iter, self.containers))
		for i in range(self.maxPerGenre):
			for iterator in iterators:
				yield(next(iterator))

	def __len__(self):
		return len(self.containers) * self.maxPerGenre

class FanFictionContainer():
	"""A container class around our fanfiction corpus. Takes stories as created by storyscraper (after they've been tar'ed)
	   and makes them easily iterable (and fast if iterated repeatedly). To do so it creates an index, which is specific
	   for a preprocessing method (as different preprocessing methods might produce different token streams).
	   You can persist the index by running the saveIndex method on the file
	   (e.g. python -c 'from util.containers import FanFictionContainer as ff; ff.saveIndex(yourFileName, yourPreprocessingmethod)')
	   If you change the preprocessing used, remove the index file!

	   Often stories contain comments by the author in the first paragraphs and the last. You can reduce the impact
	   of those comments by removing paragraphs from beginning and end in the constructor.

	   """

	log = logging.getLogger("de.t_animal.MA.util.FanFictionContainer")

	def __init__(self, file, removeTrailingParagraphs = 2, minLength = 50, preprocessing = simple_preprocess):
		""" :param files: filename of tar'ed stories as created by storyscraper. must end in tar.gz, or tar.bz, etc...
		    :param removeTrailingParagraphs: how many paragraphs from start and end to remove per story
		    :param minLength: how long (counted in tokens) a story has to be to be taken into account
		    :param preprocessing: which preprocessing step to apply when counting story lengths
		    :type file: string
		    :type removeTrailingParagraphs: int
		    :type minLength: int
		    :type preprocessing: callable"""
		self.file = file
		self.index = []
		self.removeTrailingParagraphs = removeTrailingParagraphs
		self.minLength = minLength
		self.preprocessing = preprocessing

		try:
			with gzip.open(file.rsplit(".", 2)[0] + ".index", "rt", encoding="utf-8") as indexFile:
				self.index = json.load(indexFile)
		except FileNotFoundError:
			FanFictionContainer.log.warning("No index file found, will build index on first use (might take long!). Use FanFictionContainer.save to build and persist it!")
			self.index = None

	def __iter__(self):
		"""Iterates over the stories, yielding each story as a list of tokens as returned by the chosen preprocessing method"""
		if self.index is None:
			self.index = self.buildIndex(self.file, self.preprocessing)

		tar = tarfile.open(self.file)
		nextFile = tar.next()
		while nextFile is not None:
			if not nextFile.isfile():
				continue

			storyId = nextFile.name.rsplit(".", 1)[0].split("/")[-1]

			if self._passesFilter(self.index["stories"][storyId]):
				content = json.load(io.TextIOWrapper(tar.extractfile(nextFile), "utf-8"))
				r = self.removeTrailingParagraphs
				yield [token for tokens in map(self.preprocessing, content["storytext"][r:-r]) for token in tokens]

			nextFile = tar.next()

	def __len__(self):
		if index is None:
			self.indices[indexIndex] = self.buildIndex(self.file, self.preprocessing)
			index = self.indices[indexIndex]

		return len(list(filter(self._passesFilter, index["stories"].values())))

	def _passesFilter(self, indexEntry):
		r = self.removeTrailingParagraphs
		return indexEntry["paragraphs"] > r * 2 and \
		   sum(indexEntry["paragraphLengths"][r:-r]) > self.minLength

	@staticmethod
	def buildIndex(file, preprocessing = simple_preprocess):
		index = {"storycount": 0, "stories": {}}

		tar = tarfile.open(file)
		filesDone = 0
		nextFile = tar.next()

		FanFictionContainer.log.info("Beginning building index")
		while nextFile is not None:
			if not nextFile.isfile():
				continue

			if filesDone % 10000 == 0:
				FanFictionContainer.log.info("%d files finished", filesDone)

			try:
				content = json.load(io.TextIOWrapper(tar.extractfile(nextFile), "utf-8"))
				storyId = nextFile.name.rsplit(".", 1)[0].split("/")[-1]
				processedStory = map(preprocessing, content["storytext"])
				index["stories"][storyId] = {
					"paragraphs": len(content["storytext"]),
					"paragraphLengths": list(map(len, processedStory))
				}

			except json.decoder.JSONDecodeError:
				#some files may have not been saved properly, just ignore them
				FanFictionContainer.log.debug("File %s is invalid json and is ignored", nextFile.name)
				pass

			nextFile = tar.next()
			filesDone += 1

		return index

	@staticmethod
	def saveIndex(file, preprocessing = simple_preprocess):
		"""Build and save the index of the tar'ed stories as created by storyscraper.
		:param file: the filenames of the tar'ed stories as created by storyscraper. must end in tar.gz, or tar.bz, etc...
		:param preprocessing: which preprocessing step to apply when counting story lengths
		:type file: string
		:type preprocessing: callable"""
		with gzip.open(file.rsplit(".", 2)[0] + ".index", "wt", encoding="utf-8") as indexFile:
			index = FanFictionContainer.buildIndex(file, preprocessing)
			json.dump(index, indexFile)

if __name__ == "__main__":
	logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
				level=logging.DEBUG)
	FanFictionContainer.saveIndex("/storage/MA/stories_tv.tar.gz")

class LazyModel():
	"""
		This is a wrapper around gensim model which only instantiates the wrapped
		model if (and only if) it is subscripted ([]) or a property is accessed
		(useful in conjunction with caches).
		After instantiating it, this class's object "morphs" into an object of
		the wrapped class (i.e. it casts itself and replaces its datastructures).
	"""
	log = logging.getLogger("de.t_animal.MA.util.LazyModel")

	def __init__(self, modelConstructor, *args, **kwargs):
		"""
			@param modelConstructor - this method will be called to instantiate the model (usually a constructor)
			@param *args, **kwargs - all further arguments and keyword arguments will be passed to modelConstructor
		"""

		self.modelConstructor = modelConstructor
		self.args = args
		self.kwargs = kwargs

	def _instantiate(self):
		LazyModel.log.debug("Instantiating model")
		model = self.modelConstructor(*self.args, **self.kwargs)

		LazyModel.log.debug("Morphing into model object")
		del self.modelConstructor
		del self.args
		del self.kwargs

		self.__class__ = model.__class__
		self.__dict__ = model.__dict__

	def __getattr__(self, attr):
		LazyModel.log.debug("attr %s requested, instantiating model", attr)

		if attr == "__getstate__" or attr == "__setstate__":
			#When pickled, don't instantiate first, but pickle the LazyModel object
			raise AttributeError()

		self._instantiate()
		return getattr(self, attr)

	def __getitem__(self, key):
		self._instantiate()
		return self[key]


class SubsetKeyedVectors(KeyedVectors):
	"""Builts a KeyedVectors object from a list (word, vector) tuples. This goes one step further
	   than CachedKeyedVectors, which only emulate KeyedVectors. While the latter falls back
	   to a wrapped KeyedVectors model on cache miss this class only supports the vectors which
	   were put into it.

	   This is mainly needed for supporting our cache in de Boom's RepresentationLearning
	   algorithm which relies on internals of the KeyedVector class"""

	log = logging.getLogger("de.t_animal.MA.util.SubsetKeyedVectors")

	def addToVocab(self, vocabList):
		"""Adds the vocabular in vocabList to the KeyedVectors. Very similar to KeyedVectors.load_word2vec_format.

		:param vocabList: the word, vector tuples to add
		:type vocabList: a list of tuples (word, vector) i.e. (str, numpy.array)
		"""

		vocab_size = len(vocabList)
		vector_size = vocabList[0][1].shape[0]
		datatype = vocabList[0][1].dtype

		if (self.vector_size is not None and not self.vector_size == vector_size) or \
		   (not self.syn0 == [] and self.syn0.dtype == datatype ):
			SubsetKeyedVectors.log.warn("Not adding incompatible vectors")
			return

		additionalSyn0 = numpy.zeros((vocab_size, vector_size), dtype=datatype)

		self.vector_size = vector_size
		if self.syn0 == []:
			self.syn0 = additionalSyn0
		else:
			self.syn0 = numpy.concatenate(self.syn0, additionalSyn0)

		# this class does not support count, replicate behaviour of superclass in that case
		for vocab in self.vocab.items():
			vocab.count += vocab_size

		old_vocab_size = len(self.vocab)

		for word, weights in vocabList:
			word_id = len(self.vocab)
			if word in self.vocab:
				SubsetKeyedVectors.log.warn("duplicate word '%s', ignoring all but first", word)
				continue

			# count not supported. just make up some bogus counts, in descending order
			self.vocab[word] = Vocab(index=word_id, count=old_vocab_size + vocab_size - word_id)
			self.syn0[word_id] = weights
			self.index2word.append(word)


class CachedKeyedVectors:
	""" A simple class emulating parts of gensim's KeyedVectors so that we can use their wmdistance method
		and our caching system at the same time. Not threadsafe!
	"""

	def __init__(self, model):
		"""
		:param model: the KeyedVectors model to emulate (can be a LazyModel!)
		"""
		self.model = model
		self.vocabulary = {}

	def __getitem__(self, key):
		try:
			return self.vocabulary[key]
		except KeyError:
			return self.model[key]

	def __contains__(self, key):
		return key in self.vocabulary or key in self.model

	def wmdistance(self, document1, document2):
		"""Passes the call on to gensim's KeyedVectors.wmdistance but leveraging our cache system

		:param document1: the first of the documents to get the distance of
		:type document1: a list of tuples (word token, vector)
		:param document2: the second of the documents to get the distance of
		:type document2: a list of tuples (word token, vector)
		"""

		self.vocabulary = {k:v for words in document1 + document2 for k,v in words + list(self.vocabulary.items())}

		document1Words = [word for words in document1 for word, vector in words]
		document2Words = [word for words in document2 for word, vector in words]
		return KeyedVectors.wmdistance(self, document1Words, document2Words)



class TestresultContainer():
	""" A container for test results so that they can be stored and printed
		in a consistent manner """

	log = logging.getLogger("de.t_animal.MA.util.TestresultContainer")

	def __init__(self, class1, class2, class1Label=None, class2Label=None):
		"""
		:param class1: the value expected for class one
		:param class2: the value expected for class two
		:param class1Label: the label to use for class1 (default: "class1")
		:param class2Label: the label to use for class2 (default: "class2")
		"""
		self.correct = 0
		self.incorrect = 0
		self.true_pos_class1 = 0
		self.false_pos_class1 = 0
		self.true_pos_class2 = 0
		self.false_pos_class2 = 0

		self.class1 = class1
		self.class2 = class2
		self.label1 = class1Label if class1Label is not None else "class1"
		self.label2 = class2Label if class2Label is not None else "class2"

		self.additionalData = {}

	def __getitem__(self, key):
		return self.getDict()[key]


	def addResult(self, resultingClass, expectedClass):
		""" Add a new result

		:param resultingClass: the value that was returned by the classifier
		:param expectedClass: the value that was expected
		"""
		assert(resultingClass in [self.class1, self.class2])
		assert(expectedClass in [self.class1, self.class2])

		if resultingClass == expectedClass:
			TestresultContainer.log.info("Added a correct result")
			self.correct += 1

			if expectedClass == self.class1:
				self.true_pos_class1 += 1
			else:
				self.true_pos_class2 += 1
		else:
			TestresultContainer.log.info("Added an incorrect result")
			self.incorrect += 1

			if resultingClass == self.class1:
				self.false_pos_class1 += 1
			else:
				self.false_pos_class2 += 1


	def additional(self, **kwargs):
		""" Add additional data to the container. They will be added to a dict.
		If a key exists, the corresponding value will be added to the existing
		one (i.e. lists will be appended, integers added, etc). If the value
		object does not support the + operator, the "update"-method will be called
		(so that dicts can be added, too).

		e.g.: container.additional(somekey=[4])
		container.additional(somekey=[5])
		results in a additional field like {"someKey": [4,5]}
		"""

		for key, value in kwargs.items():
			if key in self.additionalData:
				try:
					self.additionalData[key] += value
				except TypeError:
					self.additionalData[key].update(value)
			else:
				self.additionalData[key] = value


	def getDict(self):
		""" Returns a dict with information about the results. """
		total = self.correct + self.incorrect
		total_class1 = self.true_pos_class1 + self.false_pos_class2
		total_class2 = self.true_pos_class2 + self.false_pos_class1
		try:
			correct_percentage = float(self.correct) / total * 100
		except ZeroDivisionError:
			correct_percentage = float("nan")
		try:
			incorrect_percentage = float(self.incorrect) / total * 100
		except ZeroDivisionError:
			incorrect_percentage = float("nan")
		try:
			true_pos_class1_percentage = 100 * float(self.true_pos_class1) / total_class1
			false_pos_class1_percentage = 100 * float(self.false_pos_class1) / total_class1
		except ZeroDivisionError:
			true_pos_class1_percentage =  float("nan")
			false_pos_class1_percentage =  float("nan")
		try:
			true_pos_class2_percentage = 100 * float(self.true_pos_class2) / total_class2
			false_pos_class2_percentage = 100 * float(self.false_pos_class2) / total_class2
		except ZeroDivisionError:
			true_pos_class2_percentage =  float("nan")
			false_pos_class2_percentage =  float("nan")

		pvalue = scipy.stats.binom.cdf(self.incorrect, total, 0.5)

		return OrderedDict([
			("tested", total),
			("tested-" + self.label1, total_class1),
			("tested-" + self.label2, total_class2),
			("correct", self.correct),
			("correct-percentage", correct_percentage),
			("p-value", pvalue),
			("true-positive-" + self.label1, self.true_pos_class1),
			("true-positive-" + self.label2, self.true_pos_class2),
			("true-positive-" + self.label1 + "-percentage", true_pos_class1_percentage),
			("true-positive-" + self.label2 + "-percentage", true_pos_class2_percentage),
			("incorrect", self.incorrect),
			("incorrect-percentage", incorrect_percentage),
			("false-positive-" + self.label1, self.false_pos_class1),
			("false-positive-" + self.label2, self.false_pos_class2),
			("false-positive-" + self.label1 + "-percentage", false_pos_class1_percentage),
			("false-positive-" + self.label2 + "-percentage", false_pos_class2_percentage),
			("additional", self.additionalData)
		])


	def getJSON(self):
		""" Returns the dict from getDict() but formated as JSON """
		return json.dumps(self.getDict())


	def oneline(self):
		""" Returns a short oneliner describing the results"""
		vals = self.getDict()
		return "{} correct out of {} ({:6.2f}%, p={:6.3f})".format(vals["correct"], vals["tested"],
																   vals["correct-percentage"], vals["p-value"])


	def __str__(self):
		""" Returns the results in a human readable form """
		vals = self.getDict()
		padLabel1 = " " * max(0, len(self.label2) - len(self.label1))
		padLabel2 = " " * max(0, len(self.label1) - len(self.label2))
		padNoLabel = " " * max(len(self.label1), len(self.label2))
		ret = """\
				Tested:       {}   {} ({}: {}, {}: {})
				Correct:      {}   {} ({:6.2f}%)
				Incorrect:    {}   {} ({:6.2f}%)
				True positive {}: {} {:2} ({:6.2f}%)
				True positive {}: {} {:2} ({:6.2f}%)
				False positive {}:{} {:2} ({:6.2f}%)
				False positive {}:{} {:2} ({:6.2f}%)""".format(padNoLabel, vals["tested"], self.label1, vals["tested-" + self.label1],
												 self.label2, vals["tested-" + self.label2], #endTested
												 padNoLabel, vals["correct"], vals["correct-percentage"], #endCorrect
												 padNoLabel, vals["incorrect"], 100 - vals["correct-percentage"], #endIncorrect
												 self.label1, padLabel1, vals["true-positive-" + self.label1],
												 vals["true-positive-" + self.label1 + "-percentage"], #endTP1
												 self.label2, padLabel2, vals["true-positive-" + self.label2],
												 vals["true-positive-" + self.label2 + "-percentage"],  #endTP2
												 self.label1, padLabel1, vals["false-positive-" + self.label1],
												 vals["false-positive-" + self.label1 + "-percentage"], #endFP1
												 self.label2, padLabel2, vals["false-positive-" + self.label2],
												 vals["false-positive-" + self.label2 + "-percentage"])

		return textwrap.dedent(ret)

	def __lt__(self, other):
		"""Makes this class sortable by correct percentage"""
		return self.getDict()["correct-percentage"] < other.getDict()["correct-percentage"]


class CrossValidationResultContainer:

	def __init__(self, class1Label=None, class2Label=None):
		self.results = []

		self.label1 = class1Label if class1Label is not None else "class1"
		self.label2 = class2Label if class2Label is not None else "class2"

	def __getitem__(self, key):
		return self.getDict()[key]

	def addResult(self, result):
		self.results.append(result)

	def oneline(self):
		percentages = numpy.array(list(map(lambda x: x.getDict()["correct-percentage"], self.results)))

		return "Mean correct percentage: {:.3f}. Standard deviation: {:.3f}, p={:.3f})".format(percentages.mean(), percentages.std(), self.getDict()["p-value"])

	def getDict(self):
		tested = sum(map(lambda x: x.getDict()["tested"], self.results))
		tested_class1 = sum(map(lambda x: x.getDict()["tested-" + self.label1], self.results))
		tested_class2 = sum(map(lambda x: x.getDict()["tested-" + self.label2], self.results))
		correct_percentages = numpy.array(list(map(lambda x: x.getDict()["correct-percentage"], self.results)))
		incorrect_percentages = numpy.array(list(map(lambda x: x.getDict()["incorrect-percentage"], self.results)))
		true_pos_class1_percentages = numpy.array(list(map(lambda x: x.getDict()["true-positive-" + self.label1 + "-percentage"], self.results)))
		true_pos_class2_percentages = numpy.array(list(map(lambda x: x.getDict()["true-positive-" + self.label2 + "-percentage"], self.results)))
		false_pos_class1_percentages = numpy.array(list(map(lambda x: x.getDict()["false-positive-" + self.label1 + "-percentage"], self.results)))
		false_pos_class2_percentages = numpy.array(list(map(lambda x: x.getDict()["false-positive-" + self.label2 + "-percentage"], self.results)))

		pValues = [x.getDict()["p-value"] for x in self.results]
		fisher = -2 * sum(numpy.log(pValues))
		pValue = scipy.stats.ttest_1samp(correct_percentages, 50).pvalue
		combinedPValueFisher = scipy.stats.chi2.sf(fisher, 2 * len(self.results))

		return OrderedDict([
			("tested", tested),
			("tested-" + self.label1, tested_class1),
			("tested-" + self.label2, tested_class2),
			("correct-percentage-mean", correct_percentages.mean()),
			("correct-percentage-stddev", correct_percentages.std()),
			("true-positive-" + self.label1 + "-percentage-mean", true_pos_class1_percentages.mean()),
			("true-positive-" + self.label1 + "-percentage-stddev", true_pos_class1_percentages.std()),
			("true-positive-" + self.label2 + "-percentage-mean", true_pos_class2_percentages.mean()),
			("true-positive-" + self.label2 + "-percentage-stddev", true_pos_class2_percentages.std()),
			("incorrect-percentage-mean", incorrect_percentages.mean()),
			("incorrect-percentage-stddev", incorrect_percentages.std()),
			("false-positive-" + self.label1 + "-percentage-mean", false_pos_class1_percentages.mean()),
			("false-positive-" + self.label1 + "-percentage-stddev", false_pos_class1_percentages.std()),
			("false-positive-" + self.label2 + "-percentage-mean", false_pos_class2_percentages.mean()),
			("false-positive-" + self.label2 + "-percentage-stddev", false_pos_class2_percentages.std()),
			("p-value", pValue),
			("p-value-fisher", combinedPValueFisher),
			("additional", [x.getDict() for x in self.results])
		])

	def __str__(self):
		""" Returns the results in a human readable form """
		vals = self.getDict()
		padLabel1 = " " * max(0, len(self.label2) - len(self.label1))
		padLabel2 = " " * max(0, len(self.label1) - len(self.label2))
		padNoLabel = " " * max(len(self.label1), len(self.label2))
		ret = """\
				Tested:       {}  {} ({}: {}, {}: {})
				Correct:      {}   {:6.2f}% ± {:6.2f}
				Incorrect:    {}   {:6.2f}% ± {:6.2f}
				True positive {}: {} {:6.2f}% ± {:6.2f}
				True positive {}: {} {:6.2f}% ± {:6.2f}
				False positive {}:{} {:6.2f}% ± {:6.2f}
				False positive {}:{} {:6.2f}% ± {:6.2f}""".format(padNoLabel, vals["tested"], self.label1, vals["tested-" + self.label1],
												 self.label2, vals["tested-" + self.label2], #endTested
												 padNoLabel, vals["correct-percentage-mean"], vals["correct-percentage-stddev"], #endCorrect
												 padNoLabel, vals["incorrect-percentage-mean"], vals["incorrect-percentage-stddev"], #endIncorrect
												 self.label1, padLabel1, vals["true-positive-" + self.label1 + "-percentage-mean"],
												 vals["true-positive-" + self.label1 + "-percentage-stddev"], #endTP1
												 self.label2, padLabel2, vals["true-positive-" + self.label2 + "-percentage-mean"],
												 vals["true-positive-" + self.label2 + "-percentage-stddev"],  #endTP2
												 self.label1, padLabel1, vals["false-positive-" + self.label1 + "-percentage-mean"],
												 vals["false-positive-" + self.label1 + "-percentage-stddev"], #endFP1
												 self.label2, padLabel2, vals["false-positive-" + self.label2 + "-percentage-mean"],
												 vals["false-positive-" + self.label2 + "-percentage-stddev"])

		return textwrap.dedent(ret)

	def __lt__(self, other):
		"""Makes this class sortable by correct percentage mean"""
		return self.getDict()["correct-percentage-mean"] < other.getDict()["correct-percentage-mean"]