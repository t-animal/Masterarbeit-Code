#!./venv/bin/python
# PYTHON_ARGCOMPLETE_OK

import logging as log
import os
from util import isAroused
from util.classifiers import Classifier
from util.containers import TestresultContainer

from util.argGenerator import generateTrainAndValidateset, generateTestset, _dataSets, getAllFiles

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


def getClassifierClass(className, package="."):
	if not package == ".":
		package += "."

	for module in filter(lambda x: x.endswith(".py"), os.listdir(package)):
		importedModule = __import__(package + module[:-3], fromlist=[className])
		try:
			return getattr(importedModule, className)
		except AttributeError:
			pass

	raise ValueError("no such class in this package!")

def getAllClassifiers(package="."):
	for module in filter(lambda x: x.endswith(".py"), os.listdir(package)):
		for line in open(module, "r"):
			#this is only a rough estimate but good enough for now
			if line.startswith("class") and ("(SVMClassifier)" in line or "(Classifier)" in line):
				yield line[6:line.index("(")]

if __name__ == "__main__":
	import argcomplete, argparse, sys

	class ChoicesContainer:
		def __init__(self, vals):
			self.vals = list(vals) + ["all"]

		def __contains__(self, val):
			if "." in val:
				val = val[:val.index(".")]
			return val in self.vals

		def __iter__(self):
			return self.vals.__iter__()

	def ClassifierCompleter(prefix, **kwargs):
		return filter(lambda x: x.startswith(prefix), getAllClassifiers())

	parser = argparse.ArgumentParser(description='Generate train and testsets')
	parser.add_argument("--human", help="", action="store_true")
	parser.add_argument("-v", help="Be more verbose (-vv for max verbosity)", action="count", default=0)
	parser.add_argument("--train", help="", nargs="+", choices=ChoicesContainer(list(_dataSets.keys())), default=[])
	parser.add_argument("--test", help="", nargs="+", choices=ChoicesContainer(list(_dataSets.keys())))
	parser.add_argument("--validate", help="", nargs="+", choices=ChoicesContainer(list(_dataSets.keys())), default=[])
	parser.add_argument("--plot", help="Plot the vectors of a dataset", nargs="+", choices=ChoicesContainer(list(_dataSets.keys())), default=[])
	parser.add_argument("--plotFunction", help="Which plotting function to use", nargs="+", choices=["PCA", "PCA3", "LDA", "LDA3" "HIST", "MAT"], default="PCA")
	parser.add_argument("--store", help="")
	parser.add_argument("--load", help="")
	parser.add_argument("--classifier", "-c", help="", required=True).completer = ClassifierCompleter
	parser.add_argument("classifierArgs", help="additional arguments to pass to the classifier (empty to list)", nargs="*")

	argcomplete.autocomplete(parser)
	args = parser.parse_args()
	if not args.validate and not args.test:
		args.validate = args.train

	if any(map(lambda x: "all" in x, args.validate)):
		percentage = "."+args.train.split(".") if "." in args.train else ""
		args.train = list(map(lambda x: x+percentage, _dataSets.keys()))

	if any(map(lambda x: "all" in x, args.validate)):
		percentage = "."+args.validate.split(".") if "." in args.validate else ""
		args.validate = list(map(lambda x: x+percentage, _dataSets.keys()))

	if args.test and "all" in args.test:
		args.test = list(_dataSets.keys())

	if not (bool(args.train) ^ bool(args.load) or args.plot):
		parser.error("Please supply either train or load (and don't supply both) or plot")
		sys.exit(1)

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	if args.test and args.validate:
		log.warning("Supplying test sets overrides the validate sets!")


	classifierClass = getClassifierClass(args.classifier)
	trainFiles, validateFiles = generateTrainAndValidateset(args.train, args.validate)

	if len(args.classifierArgs) == 0:
		print(classifierClass.__init__.__code__.co_varnames[1:])

	classifier = classifierClass(*args.classifierArgs)
	if bool(args.train):
		classifier.train(trainFiles)
	elif bool(args.load):
		classifier.load(args.load)

	if args.store:
		classifier.store(args.store)

	result = None
	if args.test:
		testFiles = generateTestset(args.test)
		result = classifier.test(testFiles)
	elif args.validate:
		#validate the results on the validate set as if it were the test set
		result = classifier.test(validateFiles)

	if args.plot:
		if "all" in args.plot:
			args.plot = _dataSets.keys()
		for func in args.plotFunction:
			classifier.plot(getAllFiles(args.plot), func)

	if result:
		if args.human:
			if args.test:
				print("Results for training on {} ({}, {} aroused, {} not) and testing on {}".
						format(args.train, len(trainFiles), len(list(filter(isAroused, trainFiles))),
						       len(list(filter(lambda x: not isAroused(x), trainFiles))), args.test))
			else:
				print("Results for training on {} ({}, {} aroused, {} not) and validating on {}".
						format(args.train, len(trainFiles), len(list(filter(isAroused, trainFiles))),
						        len(list(filter(lambda x: not isAroused(x), trainFiles))), args.validate))

			if os.isatty(sys.stdout.fileno()):
				print("\033[1m"+result.oneline()+"\033[0m")
			else:
				print(result.oneline())
			print(result)
		else:
			print(result.getJSON())
