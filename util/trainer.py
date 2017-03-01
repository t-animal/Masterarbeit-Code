#!../venv/bin/python

import os

if __name__ == "__main__":
	from __init__ import isAroused
else:
	from util import isAroused

_dataSets = {
	"Veroff": os.path.dirname(os.path.abspath(__file__)) + "/../../infos/Veroff/",
	"Winter": os.path.dirname(os.path.abspath(__file__)) + "/../../infos/Winter_arousal_stories/",
	"Study2": os.path.dirname(os.path.abspath(__file__)) + "/../../infos/Study_2_PSE_files/"
}


def generateArguments(trainSets, testSets, testPercentage=20):
	testFiles = []
	trainFiles = []

	for testSet in testSets:
		if "." in testSet:
			testSet, percentage = testSet.split(".")
			percentage = int(percentage)
		else:
			percentage = testPercentage

		if testSet not in _dataSets:
			raise ValueError("Not a valid test set: " + testSet)

		allFiles = sorted(filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[testSet])))
		allFiles = list(map(lambda x: _dataSets[testSet] + x, allFiles))
		allAroused = list(filter(lambda x: isAroused(x), allFiles))
		allNonAroused = list(filter(lambda x: not isAroused(x), allFiles))

		testFiles += allAroused[int(-percentage * len(allFiles) / 100 / 2):]
		testFiles += allNonAroused[int(-percentage * len(allFiles) / 100 / 2):]


	for trainSet in trainSets:
		if "." in trainSet:
			trainSet, percentage = trainSet.split(".")
			percentage = int(percentage)
		else:
			percentage = 100 - testPercentage

		if trainSet not in _dataSets:
			raise ValueError("Not a valid train set: " + trainSet)

		allFiles = sorted(filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[trainSet])))
		allFiles = list(map(lambda x: _dataSets[trainSet] + x, allFiles))
		allAroused = list(filter(lambda x: isAroused(x), allFiles))
		allNonAroused = list(filter(lambda x: not isAroused(x), allFiles))

		trainFiles += filter(lambda x: x not in testFiles, allAroused[:int(percentage * len(allFiles) / 100 / 2)])
		trainFiles += filter(lambda x: x not in testFiles, allNonAroused[:int(percentage * len(allFiles) / 100 / 2)])


	#assert no testfiles are also trainfiles
	assert(set(trainFiles) - set(testFiles) == set(trainFiles))
	#assert an equal amount of aroused and non-aroused testfiles
	assert(len(list(filter(isAroused, testFiles))) == len(testFiles) / 2)

	return trainFiles, testFiles


if __name__ == "__main__":

	import argparse

	class ChoicesContainer:
		def __init__(self, vals):
			self.vals = vals

		def __contains__(self, val):
			if "." in val:
				val = val[:val.index(".")]
			return val in self.vals

		def __iter__(self):
			return self.vals.__iter__()

	parser = argparse.ArgumentParser(description='Generate train and testsets.')
	parser.add_argument("--train", help="", nargs="+", required=True, choices=ChoicesContainer(_dataSets.keys()))
	parser.add_argument("--test", help="", nargs="+", choices=ChoicesContainer(_dataSets.keys()))

	args = parser.parse_args()
	if args.test == None:
		args.test = args.train
	trainFiles, testFiles = generateArguments(args.train, args.test)

	print("--train {} --test {}".format(" ".join(trainFiles), " ".join(testFiles)))