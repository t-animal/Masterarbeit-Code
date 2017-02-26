import json
import logging
import textwrap

log = logging.getLogger("de.t_animal.MA.util.containers")

class LazyModel():
	"""
		This is a wrapper around gensim model which only instantiates the wrapped
		model if (and only if) it is subscripted ([]) or a property is accessed
		(useful in conjunction with caches).
		After instantiating it, this class's object "morphs" into an object of
		the wrapped class (i.e. it casts itself and replaces its datastructures).
	"""

	def __init__(self, modelConstructor, *args, **kwargs):
		"""
			@param modelConstructor - this method will be called to instantiate the model (usually a constructor)
			@param *args, **kwargs - all further arguments and keyword arguments will be passed to modelConstructor
		"""

		self.modelConstructor = modelConstructor
		self.args = args
		self.kwargs = kwargs

	def _instantiate(self):
		log.debug("Instantiating model")
		model = self.modelConstructor(*self.args, **self.kwargs)

		log.debug("Morphing into model object")
		del self.modelConstructor
		del self.args
		del self.kwargs

		self.__class__ = model.__class__
		self.__dict__ = model.__dict__

	def __getattr__(self, attr):
		self._instantiate()
		return getattr(self, attr)

	def __getitem__(self, key):
		self._instantiate()
		return self[key]




class TestresultContainer():
	""" A container for test results so that they can be stored and printed
	    in a consistent manner """

	def __init__(self, class1, class2, class1Label=None, class2Label=None):
		"""
		:param class1: the value expected for class one
		:param class2: the value expected for class two
		:param class1Label: the label to use for class1 (default: "class1")
		:param class2Label: the label to use for class2 (default: "class2")
		"""
		self.correct = 0
		self.incorrect = 0
		self.correct_class1 = 0
		self.incorrect_class1 = 0

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
			self.correct += 1

			if expectedClass == self.class1:
				self.correct_class1 += 1
		else:
			self.incorrect += 1

			if expectedClass == self.class1:
				self.incorrect_class1 += 1


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
		correct_percentage = float(self.correct) / total * 100
		correct_class2 = self.correct - self.correct_class1
		incorrect_class2 = self.incorrect - self.incorrect_class1
		correct_class1_percentage = 100 * float(self.correct_class1) / (self.correct_class1 + self.incorrect_class1)
		correct_class2_percentage = 100 * float(correct_class2) / (incorrect_class2 + correct_class2)

		return {
			"tested": total,
			"correct": self.correct,
			"correct-percentage": correct_percentage,
			"correct-" + self.label1: self.correct_class1,
			"correct-" + self.label2: correct_class2,
			"correct-" + self.label1 + "-percentage": correct_class1_percentage,
			"correct-" + self.label2 + "-percentage": correct_class2_percentage,
			"incorrect": self.incorrect,
			"incorrect-" + self.label1: self.incorrect_class1,
			"incorrect-" + self.label2: incorrect_class2,
			"additional": self.additionalData
		}


	def getJSON(self):
		""" Returns the dict from getDict() but formated as JSON """
		return json.dumps(self.getDict())


	def oneline(self):
		""" Returns a short oneliner describing the results"""
		vals = self.getDict()
		return "{} correct out of {} ({}%)".format(vals["correct"], vals["tested"], vals["correct-percentage"])


	def __str__(self):
		""" Returns the results in a human readable form """
		vals = self.getDict()
		ret = """\
				Tested: {}
				Correct: {} ({}%)
				Incorrect: {} ({}%)
				Correct {}: {} ({}%)
				Correct {}: {} ({}%)
				Incorrect {}: {} ({}%)
				Incorrect {}: {} ({}%)""".format(vals["tested"],
				                                 vals["correct"], vals["correct-percentage"],
				                                 vals["incorrect"], 100 - vals["correct-percentage"],
				                                 self.label1, vals["correct-" + self.label1],
				                                 vals["correct-" + self.label1 + "-percentage"],
				                                 self.label2, vals["correct-" + self.label2],
				                                 vals["correct-" + self.label2 + "-percentage"],
				                                 self.label1, vals["correct-" + self.label1],
				                                 100 - vals["correct-" + self.label1 + "-percentage"],
				                                 self.label2, vals["correct-" + self.label2],
				                                 100 - vals["correct-" + self.label2 + "-percentage"])

		return textwrap.dedent(ret)
