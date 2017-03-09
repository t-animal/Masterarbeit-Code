#!../venv/bin/python

import os

if __name__ == "__main__":
	from __init__ import isAroused
else:
	from util import isAroused

_dataSets = {
	"Veroff": os.path.dirname(os.path.abspath(__file__)) + "/../data/Veroff/",
	"Winter": os.path.dirname(os.path.abspath(__file__)) + "/../data/Winter_arousal_stories/",
	"Study2": os.path.dirname(os.path.abspath(__file__)) + "/../data/Study_2_PSE_files/",
	"Atkinson": os.path.dirname(os.path.abspath(__file__)) + "/../data/AtkinsonEtAl_nAff_cleaned/",
	"McAdams": os.path.dirname(os.path.abspath(__file__)) + "/../data/McAdams_1980_nAff_cleaned/"
}

def generateTestset(testSets):
	testFiles = []
	for testSet in testSets:
		if testSet not in _dataSets:
			raise ValueError("Not a valid test set: " + testSet)

		testFiles += map(lambda x: _dataSets[testSet] + "test/" + x,
							filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[testSet] + "test/")))

	return testFiles

def generateTrainAndValidateset(trainSets, validateSets, validatePercentage=20):
	validateFiles = []
	trainFiles = []

	for validateSet in validateSets:
		if "." in validateSet:
			validateSet, percentage = validateSet.split(".")
			percentage = int(percentage)
		else:
			percentage = validatePercentage

		if validateSet not in _dataSets:
			raise ValueError("Not a valid validate set: " + validateSet)

		allFiles = sorted(filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[validateSet])))
		allFiles = list(map(lambda x: _dataSets[validateSet] + x, allFiles))
		allAroused = list(filter(lambda x: isAroused(x), allFiles))
		allNonAroused = list(filter(lambda x: not isAroused(x), allFiles))

		validateFiles += allAroused[int(-percentage * len(allFiles) / 100 / 2):]
		validateFiles += allNonAroused[int(-percentage * len(allFiles) / 100 / 2):]


	for trainSet in trainSets:
		if "." in trainSet:
			trainSet, trainPercentage = trainSet.split(".", 1)
			if "." in trainPercentage:
				trainPercentage, validatePercentage = map(int, trainPercentage.split("."))
			trainPercentage = int(trainPercentage)
		else:
			trainPercentage = 100 - validatePercentage
			validatePercentage = validatePercentage

		if trainSet not in _dataSets:
			raise ValueError("Not a valid train set: " + trainSet)

		allFiles = sorted(filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[trainSet])))
		allFiles = list(map(lambda x: _dataSets[trainSet] + x, allFiles))
		allAroused = list(filter(lambda x: isAroused(x), allFiles))
		allNonAroused = list(filter(lambda x: not isAroused(x), allFiles))

		trainFiles += filter(lambda x: x not in validateFiles, allAroused[:int(trainPercentage * len(allFiles) / 100 / 2)])
		trainFiles += filter(lambda x: x not in validateFiles, allNonAroused[:int(trainPercentage * len(allFiles) / 100 / 2)])

	#assert no validatefiles are also trainfiles
	assert(set(trainFiles) - set(validateFiles) == set(trainFiles))
	#assert an equal amount of aroused and non-aroused validatefiles
	assert(len(list(filter(isAroused, validateFiles))) == len(validateFiles) / 2)

	return trainFiles, validateFiles


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

	parser = argparse.ArgumentParser(description='Generate train and validatesets.')
	parser.add_argument("--train", help="", nargs="+", required=True, choices=ChoicesContainer(_dataSets.keys()))
	parser.add_argument("--validate", help="", nargs="+", choices=ChoicesContainer(_dataSets.keys()))

	args = parser.parse_args()
	if args.validate == None:
		args.validate = args.train
	trainFiles, validateFiles = generateTrainAndValidateset(args.train, args.validate)

	print("--train {} --validate {}".format(" ".join(trainFiles), " ".join(validateFiles)))