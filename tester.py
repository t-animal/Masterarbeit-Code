#!./venv/bin/python
# PYTHON_ARGCOMPLETE_OK

import logging as log
import os
from util import isAroused
from util.classifiers import Classifier
from util.containers import CrossValidationResultContainer, TestresultContainer

from util.argGenerator import getAllFiles, \
                              generateCrossValidationSets, \
                              generateTrainAndValidateset, \
                              generateTestset, \
                              _dataSets

class CacheGenerator(Classifier):
	"""This is a short class designed to be run once per model on all datasets
		to pre-fill the vector cache"""
	def train(self, trainFiles):
		for filename in trainFiles:
			log.info("Caching file vectors of %s", filename)
			self._vectorizeFile(filename)
		return TestresultContainer(True, False, "", "")

	def test(self, trainFiles):
		for filename in trainFiles:
			log.info("Caching file vectors of %s", filename)
			self._vectorizeFile(filename)
		return TestresultContainer(True, False, "", "")


class ChoicesContainer:
	def __init__(self, vals):
		self.vals = list(vals) + ["all"]

	def __contains__(self, val):
		if "." in val:
			val = val[:val.index(".")]
		return val in self.vals

	def __iter__(self):
		return self.vals.__iter__()

def classifierCompleter(prefix, **kwargs):
	try:
		return list(filter(lambda x: x.startswith(prefix), getAllClassifiers("classifiers")))
	except Exception as e:
		return [str(e)]


def getClassifierClass(className, package="."):
	packagePath = package
	if not package == ".":
		package += "."

	for module in filter(lambda x: x.endswith(".py"), os.listdir(packagePath)):
		importedModule = __import__(package + module[:-3], fromlist=[className])
		try:
			return getattr(importedModule, className)
		except AttributeError:
			pass

	raise ValueError("no such class in this package!")

def getAllClassifiers(package="."):
	for module in filter(lambda x: x.endswith(".py"), os.listdir(package)):
		for line in open(os.path.join(package, module), "r"):
			#this is only a rough estimate but good enough for now
			if line.startswith("class") and ("SVMClassifier" in line or "Classifier" in line):
				yield line[6:line.index("(")]
				break #assuming one classifier per file


def trainAndTest(classifier, train, test, load=None, store=None):
	"""
		Trains a given classifier on the given trainSets and tests it on the given testSets
	"""

	if load is None:
		classifier.train(train)
	else:
		classifier.load(load)

	if store is not None:
		classifier.store(store)

	return classifier.test(test)

def checkForAll(dataSetsList):
	"""Returns all available datasets if the dataSetsList is an list and it contains
	the word "all" otherwise returns the dataSetsList unchanged"""
	if type(dataSetsList) is not list:
		return dataSetsList

	for dataSet in dataSetsList:
		dataSet = dataSet.split(".")[0]
		if dataSet == "all":
			percentage = "." + dataSet.split(".")[1] if "." in dataSet else ""
			return list(map(lambda x: x + percentage, _dataSets.keys()))

	return dataSetsList

def printResult(result, json, train, trainFiles, validate, test=None):
	if not json:
		if test is not None:
			print("Results for training on {} ({}, {} aroused, {} not) and testing on {}".
					format(train, len(trainFiles), len(list(filter(isAroused, trainFiles))),
					       len(list(filter(lambda x: not isAroused(x), trainFiles))), test))
		else:
			print("Results for training on {} ({}, {} aroused, {} not) and validating on {}".
					format(train, len(trainFiles), len(list(filter(isAroused, trainFiles))),
					        len(list(filter(lambda x: not isAroused(x), trainFiles))), validate))

		if os.isatty(sys.stdout.fileno()):
			print("\033[1m"+result.oneline()+"\033[0m")
		else:
			print(result.oneline())
		print(result)
	else:
		print(result.getJSON())

if __name__ == "__main__":
	import argcomplete, argparse, configparser, sys

	configParser = configparser.ConfigParser()
	configParser.read(os.path.split(__file__)[0] + os.sep + "tester.ini")
	if "ModelPaths" in configParser:
		modelPaths = dict(configParser["ModelPaths"])
	else:
		modelPaths = {}

	class ArgSplit(argparse.Action):
		def __init__(self, option_strings, dest, **kwargs):
			self.dest = dest
			super(ArgSplit, self).__init__(option_strings, dest, **kwargs)

		def __call__(self, parser, namespace, values, option_string=None):

			if any(["=" not in v for v in values]):
				raise ValueError("Args must be key value pairs, joined by a = (key=value)")

			setattr(namespace, self.dest, dict([val.split("=", 1) for val in values]))

	parser = argparse.ArgumentParser(description='Train, validate and test classifiers')
	parser.add_argument("--json",  help = "Display the output as json",              action = "store_true")
	parser.add_argument("-v",      help = "Be more verbose (-vv for max verbosity)", action = "count", default = 0)

	parser.add_argument("--train", help = "The datasets to train on",
	                               nargs = "+", choices = ChoicesContainer(_dataSets.keys()), default = [])
	parser.add_argument("--test",  help = "The datasets to test on (overrides any validation sets!)",
	                               nargs = "+", choices = ChoicesContainer(_dataSets.keys()))
	parser.add_argument("--validate", help = "The datasets to validate on",
	                               nargs = "+", choices = ChoicesContainer(_dataSets.keys()), default = [])

	parser.add_argument("--crossValidate", help = "The datasets to crossvalidate on",
	                               nargs = "+", choices = ChoicesContainer(_dataSets.keys()), default = [])

	parser.add_argument("--plot",  help = "Plot the vectors of a dataset",
	                               nargs = "+", choices = ChoicesContainer(_dataSets.keys()), default = [])
	parser.add_argument("--plotFunction", help = "Which plotting function to use",
	                               nargs = "+", choices = ["PCA", "PCA3", "LDA", "LDA3" "HIST", "MAT"], default = "PCA")

	parser.add_argument("--store", help = "Store the trained classifier to this path")
	parser.add_argument("--load",  help = "Load a pre-trained classifier from this path")

	parser.add_argument("--classifier", "-c", help = "Which classifier to use", required = True) \
	                   .completer = classifierCompleter
	parser.add_argument("--modelPath", "-m",  help = "Load the model path from 'tester.ini' and pass it to the classifier",
	                               choices = list(modelPaths.keys()))
	parser.add_argument("--args",  help = "additional arguments to pass to the classifier as key=value pairs",
	                               nargs = "*", dest="classifierArgs", action=ArgSplit, default={})

	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	#sanitize the arguments
	if not args.validate and not args.test:
		args.validate = args.train

	if args.modelPath:
		args.classifierArgs["modelPath"] = modelPaths[args.modelPath]

	args.train = checkForAll(args.train)
	args.validate = checkForAll(args.validate)
	args.test = checkForAll(args.test)

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	#Perform sanity checks on the arguments
	if not (bool(args.train) ^ bool(args.load) or args.plot or args.crossValidate or args.crossTest):
		parser.error("Please supply either train or load (and don't supply both) or plot")
		sys.exit(1)

	if args.load and not (args.validate or args.test):
		parser.error("Loading a stored classifier and not supplying validation or testing!")
		sys.exit(1)

	if args.test and args.validate:
		log.warning("Supplying test sets overrides the validate sets!")



	#actual computation begins here
	classifierClass = getClassifierClass(args.classifier, "classifiers")
	classifier = classifierClass(**args.classifierArgs)

	if args.plot:
		if "all" in args.plot:
			args.plot = _dataSets.keys()
		for func in args.plotFunction:
			classifier.plot(getAllFiles(args.plot), func)

	if args.train or args.load:
		trainFiles, validateFiles = generateTrainAndValidateset(args.train, args.validate)

		if args.test:
			testFiles = generateTestset(args.test)
		else:
			testFiles = validateFiles

		result = trainAndTest(classifier, trainFiles, testFiles, args.load, args.store)
		printResult(result, args.json, args.train, trainFiles, args.validate, args.test)


	if args.crossValidate:
		crossSet = generateCrossValidationSets(args.crossTest)

		result = CrossValidationResultContainer("aroused", "nonAroused")

		for crossTestSet in crossSet:
			trainSet = crossTestSet["crossValidate"][0]["train"] + crossTestSet["crossValidate"][0]["validate"]
			result.addResult(trainAndTest(classifier, trainSet, crossTestSet["outerValidate"]))

		printResult(result, args.json, args.crossTest, trainSet, None, args.crossTest)
