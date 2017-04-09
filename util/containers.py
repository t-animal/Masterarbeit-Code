import json
import logging
import numpy
import scipy
import scipy.stats
import textwrap

from collections import OrderedDict


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
		combinedPValue = scipy.stats.chi2.sf(fisher, 2 * len(self.results))

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
			("p-value", combinedPValue),
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