#!../venv/bin/python

import os
import random

if __name__ == "__main__":
	from __init__ import isAroused
else:
	from util import isAroused

_dataSets = {
	"Veroff": os.path.dirname(os.path.abspath(__file__)) + "/../data/Veroff/",
	"Winter": os.path.dirname(os.path.abspath(__file__)) + "/../data/Winter_arousal_stories/",
	"Study2": os.path.dirname(os.path.abspath(__file__)) + "/../data/Study_2_PSE_files/",
	"Atkinson": os.path.dirname(os.path.abspath(__file__)) + "/../data/AtkinsonEtAl_nAff_cleaned/",
	"McAdams": os.path.dirname(os.path.abspath(__file__)) + "/../data/McAdams_1980_nAff_cleaned/",
	"McClelland": os.path.dirname(os.path.abspath(__file__)) + "/../data/McClelland_et_al_nAch_cleaned/",
	"Wirth": os.path.dirname(os.path.abspath(__file__)) + "/../data/PSE_WirthSchultheiss2006/"
}

def getAllFiles(dataSets):
	"""
	Returns a list of all files in the data sets contained in dataSets. That is, all txt files
	in the directory (and the test directory within that) of the data sets.

	:param dataSets: the data sets to get the files of
	:type dataSets: list of strings (entries must be in _dataSets.keys())
	"""

	files = []
	for dataSet in dataSets:
		if dataSet not in _dataSets:
			raise ValueError("Not a valid test set: " + dataSet)

		files += map(lambda x: _dataSets[dataSet] + "test/" + x,
							filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[dataSet] + "test/")))
		files += map(lambda x: _dataSets[dataSet] + x,
							filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[dataSet])))

	return files

def generateCrossValidationSets(dataSets):
	"""Generates cross validation sets. I.e. all files from the given data sets (files in the
	directory of the data set and in the test directory within that) are partitioned into 5 sets
	of 20% test data, and 80% training data. These 80% training data are once more partitioned into
	N sets of 80% training and 20% validation data. That yields a structure like this:
	[
		[
			[Training Data(0.64% of total data)],
			[ValidationData(0.16% of total data)]
		]*5
		[Testing Data(0.2% of total data)]
	]*5

	This is done seperately on all data sets to ensure an equal distribution across training, validation
	and testing data. Within the testing and validation data an equal number of aroused and
	non-aroused files are guaranteed. It is also guaranteed that no training files are in the validation
	or testing data.
	"""

	embeddedCrossvalidationSets = []
	for dataSet in dataSets:

		allFiles = getAllFiles([dataSet])
		allAroused = list(filter(lambda x: isAroused(x), allFiles))
		allNonAroused = list(filter(lambda x: not isAroused(x), allFiles))

		random.seed(42)
		random.shuffle(allAroused)
		random.shuffle(allNonAroused)

		for crossTestingI in range(0, 5):
			if len(embeddedCrossvalidationSets) <= crossTestingI:
				embeddedCrossvalidationSets += [{"test": [], "crossValidate": []}]

			crossTestingSet = embeddedCrossvalidationSets[crossTestingI]

			testingAroused = allAroused[crossTestingI::5]
			testingNonAroused = allNonAroused[crossTestingI::5]

			testingAroused = testingAroused[:len(testingNonAroused)]
			testingNonAroused = testingNonAroused[:len(testingAroused)]

			testingSet = testingAroused + testingNonAroused
			restAroused = list(filter(lambda x: x not in testingSet, allAroused))
			restNonAroused = list(filter(lambda x: x not in testingSet, allNonAroused))

			assert(len(list(filter(isAroused, testingSet))) == len(testingSet) / 2)
			crossTestingSet["test"] += testingSet

			for crossValidateI in range(0, 5):
				if len(crossTestingSet["crossValidate"]) <= crossValidateI:
					crossTestingSet["crossValidate"] += [{"validate": [], "train": []}]

				crossValidationSet = crossTestingSet["crossValidate"][crossValidateI]

				validatingAroused = restAroused[crossValidateI::5]
				validatingNonAroused = restNonAroused[crossValidateI::5]

				validatingAroused = validatingAroused[:len(validatingNonAroused)]
				validatingNonAroused = validatingNonAroused[:len(validatingAroused)]

				validatingSet = validatingAroused + validatingNonAroused
				trainingSet = list(filter(lambda x: x not in validatingSet, restAroused)) + \
				              list(filter(lambda x: x not in validatingSet, restNonAroused))

				assert(len(list(filter(isAroused, validatingSet))) == len(validatingSet) / 2)
				#assert no validate files or testing files are train files
				assert(set(trainingSet) - set(validatingSet) == set(trainingSet))
				assert(set(trainingSet) - set(testingSet) == set(trainingSet))

				crossValidationSet["validate"] += validatingSet
				crossValidationSet["train"] += trainingSet

	return embeddedCrossvalidationSets



def generateTestset(testSets):
	"""
	Returns a list of all test files in the data sets contained in dataSets. That is, a list of
	all txt files in a directory "test" within the directory of the data set.

	:param dataSets: the data sets to get the test files of
	:type dataSets: list of strings (entries must be in _dataSets.keys())
	"""
	testFiles = []
	for testSet in testSets:
		if testSet not in _dataSets:
			raise ValueError("Not a valid test set: " + testSet)

		testFiles += map(lambda x: _dataSets[testSet] + "test/" + x,
							filter(lambda x: x.endswith(".txt"), os.listdir(_dataSets[testSet] + "test/")))

	return testFiles

def generateTrainAndValidateset(trainSets, validateSets, validatePercentage=20):
	"""
	Returns two lists, both with filenames taken from the training data. The first list contains
	filenames to train upon, the second list contains filenames to test upon. It is guaranteed (1)
	that no entries from the training list are contained in the testing list. It is guaranteed (2)
	that in the testing list are exactly as many aroused as non aroused files. The files are taken
	from the data sets in trainSets and validateSets. Those do not have to be the same datasets.

	The values of trainSets and validateSets must be in _dataSets.keys(). They may contain a dot (.)
	followed by a number N (e.g. ["Veroff.65"]). In this case the function assigns N percent of the
	available files to the respective list. If the percentages cannot be fulfilled without breaking
	guarantee (1) the validateSet are given priority (the percentage of files are assigned
	first, i.e. passing a validate set with 100% and the same set also as a training set will result
	in an empty training set). If no percentages are given, the sets will be partioned into 80%
	training and 20% validate sets.

	:param dataSets: the data sets to get the training files of
	:type dataSets: list of strings (entries must be in _dataSets.keys()), may contain . (see above)
	:param validateSets: the data sets to get the validating files of
	:type validateSets: list of strings (entries must be in _dataSets.keys()), may contain . (see above)
	:param validatePercentage: the data sets to get the training files of
	:type validatePercentage: int
	"""
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
		random.seed(42) #make sure all lists are randomized equally each time
		random.shuffle(allFiles)

		allAroused = list(filter(lambda x: isAroused(x), allFiles))
		allNonAroused = list(filter(lambda x: not isAroused(x), allFiles))

		validateFiles += allAroused[len(allAroused) - int(percentage * len(allFiles) / 100 / 2):]
		validateFiles += allNonAroused[len(allNonAroused) - int(percentage * len(allFiles) / 100 / 2):]


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
		random.seed(42) #make sure all lists are randomized equally each time
		random.shuffle(allFiles)

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