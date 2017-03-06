#!./venv/bin/python

import logging as log
import os

from util.argGenerator import generateArguments, _dataSets

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


if __name__ == "__main__":
	import argparse, sys

	class ChoicesContainer:
		def __init__(self, vals):
			self.vals = list(vals) + ["all"]

		def __contains__(self, val):
			if "." in val:
				val = val[:val.index(".")]
			return val in self.vals

		def __iter__(self):
			return self.vals.__iter__()

	parser = argparse.ArgumentParser(description='Generate train and testsets')
	parser.add_argument("--human", help="", action="store_true")
	parser.add_argument("-v", help="Be more verbose (-vv for max verbosity)", action="count", default=0)
	parser.add_argument("--train", help="", nargs="+", choices=ChoicesContainer(list(_dataSets.keys())), default=[])
	parser.add_argument("--test", help="", nargs="+", choices=ChoicesContainer(list(_dataSets.keys())))
	parser.add_argument("--store", help="")
	parser.add_argument("--load", help="")
	parser.add_argument("--classifier", "-c", help="", required=True)
	parser.add_argument("classifierArgs", help="additional arguments to pass to the classifier (empty to list)", nargs="*")

	args = parser.parse_args()
	if args.test == None:
		args.test = args.train

	if "all" in args.train:
		percentage = "."+args.train.split(".") if "." in args.train else ""
		args.train = list(map(lambda x: x+percentage, _dataSets.keys()))

	if "all" in args.test:
		percentage = "."+args.test.split(".") if "." in args.test else ""
		args.test = list(map(lambda x: x+percentage, _dataSets.keys()))

	if not (bool(args.train) ^ bool(args.load)):
		parser.error("Please supply either train or load and don't supply both")
		sys.exit(1)

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[log.WARN, log.INFO, log.DEBUG][min(args.v, 2)])

	classifierClass = getClassifierClass(args.classifier)
	trainFiles, testFiles = generateArguments(args.train, args.test)

	if len(args.classifierArgs) == 0:
		print(classifierClass.__init__.__code__.co_varnames[1:])

	classifier = classifierClass(*args.classifierArgs)
	if bool(args.train):
		classifier.train(trainFiles)
	else:
		classifier.load(args.load)

	if args.store:
		classifier.store(args.store)

	result = classifier.test(testFiles)

	if args.human:
		print("Results for training on {} and testing on {}".format(args.train, args.test))
		if os.isatty(sys.stdout.fileno()):
			print("\033[1m"+result.oneline()+"\033[0m")
		else:
			print(result.oneline())
		print(result)
	else:
		print(result.getJSON())
