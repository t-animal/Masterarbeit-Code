#!../venv/bin/python

"""
This file is an adaption of our cross validation code for the Winter baseline calculations
"""

import random

from containers import *
from itertools import filterfalse


def isAroused(storyId):
	return "aroused" in storyId

def getAllStories(dataSets):
	stories = []
	for dataSet in dataSets:
		for author in _resultDict[dataSet]:
			stories += [dataSet + "__" + author + "__" + str(index) for index, _ in enumerate(_resultDict[dataSet][author])]
	return stories

def generateCrossValidationSets(dataSets, shuffleSeed=42):
	"""Generates cross validation sets. I.e. all stories from the given data sets are partitioned into 5 sets
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
		allFiles = getAllStories([dataSet])
		allAroused = list(filter(lambda x: isAroused(x), allFiles))
		allNonAroused = list(filter(lambda x: not isAroused(x), allFiles))

		random.seed(shuffleSeed)
		random.shuffle(allAroused)
		random.shuffle(allNonAroused)

		for outerIndex in range(0, 5):
			if len(embeddedCrossvalidationSets) <= outerIndex:
				embeddedCrossvalidationSets += [{"outerValidate": [], "crossValidate": []}]

			outerSet = embeddedCrossvalidationSets[outerIndex]

			outerAroused = allAroused[outerIndex::5]
			outerNonAroused = allNonAroused[outerIndex::5]

			outerAroused = outerAroused[:len(outerNonAroused)]
			outerNonAroused = outerNonAroused[:len(outerAroused)]

			outerValidateSet = outerAroused + outerNonAroused
			restAroused = list(filter(lambda x: x not in outerValidateSet, allAroused))
			restNonAroused = list(filter(lambda x: x not in outerValidateSet, allNonAroused))

			assert(len(list(filter(isAroused, outerValidateSet))) == len(outerValidateSet) / 2)
			outerSet["outerValidate"] += outerValidateSet

			for innerIndex in range(0, 5):
				if len(outerSet["crossValidate"]) <= innerIndex:
					outerSet["crossValidate"] += [{"validate": [], "train": []}]

				crossValidationSet = outerSet["crossValidate"][innerIndex]

				validatingAroused = restAroused[innerIndex::5]
				validatingNonAroused = restNonAroused[innerIndex::5]

				validatingAroused = validatingAroused[:len(validatingNonAroused)]
				validatingNonAroused = validatingNonAroused[:len(validatingAroused)]

				validatingSet = validatingAroused + validatingNonAroused
				trainingSet = list(filter(lambda x: x not in validatingSet, restAroused)) + \
				              list(filter(lambda x: x not in validatingSet, restNonAroused))

				assert(len(list(filter(isAroused, validatingSet))) == len(validatingSet) / 2)
				#assert no validate files or testing files are train files
				assert(set(trainingSet) - set(validatingSet) == set(trainingSet))
				assert(set(trainingSet) - set(outerValidateSet) == set(trainingSet))

				crossValidationSet["validate"] += validatingSet
				crossValidationSet["train"] += trainingSet

	return embeddedCrossvalidationSets

class ThresholdingClassifier():

	def _getWinterImageCount(self, storyId):
			dataSet, author, index = storyId.split("__")
			return _resultDict[dataSet][author][int(index)]


	def train(self, storyIds):
		bestThreshold = -1
		bestCorrect = -1
		weightAroused = len(storyIds)/2/len(list(filter(isAroused, storyIds)))
		weightControl = len(storyIds)/2/len(list(filterfalse(isAroused, storyIds)))

		#going in .5 steps means we don't have to care
		for threshold in range(0,15):
			curCorrect = 0
			for storyId in storyIds:
				result = self._getWinterImageCount(storyId) > threshold
				if result == isAroused(storyId):
					curCorrect += weightAroused if isAroused(storyId) else weightControl

			if curCorrect > bestCorrect:
				bestThreshold = threshold
				bestCorrect = curCorrect

		self.threshold = bestThreshold


	def test(self, storyIds):
		testResult = TestresultContainer(True, False, "aroused", "nonAroused")

		for storyId in storyIds:
			result = self._getWinterImageCount(storyId) > self.threshold
			testResult.addResult(result, isAroused(storyId))

		return testResult


def performCV(dataSets, reshuffle = True):
	classifier = ThresholdingClassifier()

	generatedSets = generateCrossValidationSets(dataSets, 42)
	if reshuffle:
		generatedSets += generateCrossValidationSets(dataSets, 23)

	result = CrossValidationResultContainer("aroused", "nonAroused")

	for crossValidateSet in generatedSets:
		trainSet = crossValidateSet["crossValidate"][0]["train"] + crossValidateSet["crossValidate"][0]["validate"]
		classifier.train(trainSet)
		result.addResult(classifier.test(crossValidateSet["outerValidate"]))

	return result

_resultDict = \
{"Atkinson":{"aroused01R18":[1,0,2,0,0,1],"control02N15":[1,0,0,3,0,2],"aroused02R20":[2,1,0,1,0,0],"control03N34":[0,0,3,1,0,0],"control06N13":[0,1,0,2,2,0],"aroused06R02":[2,0,1,0,2,0],"control10N32":[0,1,0,1,0,1],"control11N09":[1,0,0,1,1,3],"aroused11R19":[1,1,2,3,3,0],"control16N07":[1,0,0,2,0,2],"aroused17R29":[2,2,0,2,0,2],"aroused19R22":[0,2,0,2,1,1],"aroused22R03":[2,0,2,1,1,0],"aroused24R31":[1,0,2,1,2,2],"control26N16":[0,1,0,2,2,0],"aroused26R14":[0,1,1,1,2,0],"control30N05":[1,0,0,3,3,3],"aroused31R23":[2,0,3,2,2,0],"control32N02":[0,2,0,0,2,0],"control36N11":[1,2,0,0,0,0],"control37N35":[1,4,0,0,2,0],"control38N18":[0,0,4,3,0,0],"control39N25":[1,0,1,2,2,0],"aroused39R17":[1,0,0,0,0,0],"control44N17":[4,3,2,3,2,0],"aroused45R15":[1,0,0,1,2,0],"aroused46R16":[0,0,0,0,1,0],"aroused46R27":[0,0,3,0,2,1],"aroused47R28":[1,0,1,1,1,0],"control48N26":[2,0,0,1,3,0],"control51N01":[2,3,2,2,4,2],"aroused53R21":[3,1,2,0,1,3],"control54N36":[0,0,0,0,3,1],"control59N24":[0,0,0,3,0,0],"control59N29":[0,2,0,1,0,0],"aroused60R24":[3,2,0,0,0,0],"control61N23":[4,0,3,0,1,0],"aroused61R12":[2,1,0,1,0,0],"control62N22":[2,2,0,0,0,0],"aroused64R13":[1],"control65N08":[3,0,4,0,0,0],"control66N20":[1,2,0,3,3,0],"control68N03":[3,2,0,0,2,2],"control69N31":[2,0,3,2,2,0],"aroused69R04":[2,2,0,2,3,0],"control70N04":[2,1,0,0,0,1],"control71N10":[0,1,0,2,4,0],"aroused74R26":[0,1,1,1,2,1],"aroused78R07":[2,0,0,3,1,2],"aroused80R06":[3,3,1,1,3,0],"control81N19":[2,0,0,3,1,0],"control82N12":[0,0,0,2,0,0],"control84N06":[0,1,0,5,4,2],"aroused85R08":[0,0,0,0,1,0],"control91N27":[0,0,1,0,0,1],"aroused91R25":[1,0,0,2,0,0],"control92N14":[0,0,0,1,2,1],"aroused92R09":[0,1,0,2,0,2],"control93N33":[0,1,0,2,1,3],"aroused94R05":[0,2,0,0,2,1],"control97N28":[1,0,3,2,0,0],"control97N30":[2,0,0,2,2,1],"aroused98R10":[0,0,3,1,0,0],"aroused98R30":[4,1,1,1,0,0],"control99N21":[0,0,0,3,2,0]},
"McAdams": {"aroused001": [2,0,1,2],"aroused002": [1,1,2,1],"aroused003": [2,0,0,2],"aroused004": [0,0,0,0],"aroused006": [1,1,1,1],"aroused007": [2,1,1,1],"aroused008": [2,0,0,3],"control009": [2,0,1,2],"aroused005": [3,1,2,4],"control010": [0,0,0,0],"control011": [2,0,1,0],"control012": [2,0,2,1],"aroused013": [1,0,0,1],"aroused014": [0,0,1,1],"aroused015": [1,2,1,0],"aroused016": [2,0,1,2],"aroused017": [3,0,1,3],"aroused018": [2,0,1,1],"aroused019": [1,1,1,0],"aroused032": [2,2,2,2],"aroused033": [2,0,1,2],"aroused034": [2,1,1,1],"aroused035": [3,0,3,2],"aroused036": [1,0,1,2],"aroused038": [3,0,0,1],"aroused020": [1,1,2,2],"aroused021": [3,2,2,1],"aroused022": [1,1,2,1],"aroused023": [1,2,1,0],"aroused024": [1,2,1,1],"aroused025": [2,2,3,1],"aroused026": [2,1,2,2],"aroused027": [1,0,2,2],"aroused028": [2,2,3,2],"aroused040": [1,0,1,0],"aroused041": [1,0,1,2],"aroused042": [0,0,1,1],"aroused043": [1,1,1,0],"aroused044": [2,0,0,2],"aroused046": [1,1,1,2],"aroused048": [3,0,2,2],"aroused049": [2,0,1,1],"aroused050": [3,0,1],"aroused055": [3,2,2,1],"aroused056": [3,0,1,2],"aroused063": [0,0,0,1],"control085": [2,1,1,2],"control086": [1,1,1,2],"control087": [1,3,2,3],"control088": [1,0,0,1],"control089": [0,0,1,0],"control090": [1,0,2,0],"control091": [3,2,1,3],"control092": [2,0,2,1],"control093": [1,1,2,1],"control094": [1,2,2,1],"control095": [2,2,2,3],"control096": [1,1,1,1],"control097": [3,1,1,2],"control098": [1,1,0,2],"control099": [2,2,2,2],"control100": [0,2,0,0],"control101": [0,0,2,0],"control102": [1,1,2,2],"control103": [0,0,1,0],"control104": [0,0,1,1],"control105": [2,0,2,2],"control106": [2,0,1,2],"control107": [0,0,1,1],"control108": [0,1,1,0],"control109": [1,0,0,1],"control110": [1,0,1,1],"control111": [1,1,1,2],"control112": [2,2,1,3],"control113": [2,0,3,2],"control114": [1,1,1,2],"control115": [1,0,1,2],"control116": [1,0,0,1],"control117": [3,0,1,0],"control118": [3,0,2,1],"control119": [2,0,0,2],"control120": [1,1,1,2],"control121": [2,0,1,1],"control122": [1,0,0,0]},
"McClelland":{"control01E33":[3,3,0,2],"control02E01":[0,0,0,0],"aroused02U21":[2,0,0,1],"aroused03A12":[6,2,1,3],"aroused03U03":[1,1,0,1],"aroused04A14":[0,1,0,0],"aroused06U25":[3,0,0,1],"aroused06U29":[1,0,0,0],"aroused07U26":[2,0,0,0],"aroused08U31":[1,0,0,0],"aroused09A02":[0,0,0,0],"aroused10A16":[0,0,1,2],"aroused10U11":[5,0,3,0],"control11E02":[0,1,0,0],"control11E07":[0,0,0,0],"control11E36":[1,0,0,0],"aroused13U13":[3,3,0,1],"aroused14A36":[3,4,4,0],"aroused15U02":[5,0,3,0],"aroused15U08":[2,2,2,3],"aroused15U12":[0,0,1,0],"control17E08":[0,1,0,0],"aroused18U22":[2,0,0,0],"aroused18U37":[3,0,0,0],"aroused19U18":[7,0,0,4],"aroused21A07":[0,0,0,1],"aroused21A17":[0,0,0,0],"control21E32":[0,3,0,0],"control22E39":[0,0,0,0],"aroused23A31":[0,2,2,0],"aroused24A30":[5,2,1,1],"aroused25A21":[1,0,2,1],"aroused25U19":[4,0,4,5],"aroused26A38":[1,1,1,4],"control26E16":[0,0,0,0],"aroused27A04":[0,0,0,0],"control27E04":[0,0,1,4],"control29E21":[1,0,1,2],"aroused29U01":[0,0,0,2],"aroused29U06":[0,0,0,4],"control30E31":[0,0,0,0],"aroused30U05":[1,0,0,1],"aroused31A27":[5,0,0,1],"control32E05":[0,0,0,0],"aroused33A37":[0,1,3,0],"control34E29":[0,0,0,0],"aroused35A13":[4,0,0,1],"control36E19":[0,2,0,0],"control36E24":[4,0,1,0],"aroused36U09":[4,0,0,0],"aroused36U39":[2,0,0,1],"control38E37":[0,0,1,0],"aroused39A28":[4,0,4,0],"control39E30":[0,0,0,0],"aroused39U28":[1,1,0,3],"control40E15":[0,3,0,3],"aroused40U10":[0,2,0,2],"aroused43U27":[3,2,3,1],"aroused45A11":[4,1,1,0],"aroused47A23":[0,1,3,0],"control48E03":[1,0,1,0],"aroused49A06":[0,1,0,1],"aroused49U35":[3,0,1,1],"aroused51A19":[3,0,0,1],"control51E14":[0,1,0,0],"control54E23":[0,3,0,0],"aroused54U34":[2,3,2,0],"aroused55U20":[5,1,1,4],"aroused56A24":[0,2,1,0],"control59E35":[4,0,0,2],"aroused60A05":[4,2,3,0],"control62E09":[4,2,0,0],"aroused62U24":[0,0,1,0],"aroused62U33":[2,0,2,1],"control63E18":[0,0,0,0],"aroused64A18":[1,0,0,0],"control66E06":[1,0,0,0],"aroused66U14":[2,0,0,0],"aroused67U16":[1,2,3,3],"aroused70A32":[3,1,0,0],"control73E26":[1,0,0,0],"aroused74A26":[0,0,0,0],"aroused75A03":[1,3,2,0],"control75E10":[0,0,0,0],"control75E22":[0,0,0,1],"aroused76A10":[1,0,0,0],"aroused76U17":[1,2,0,0],"aroused76U36":[1,0,0,0],"control77E27":[2,1,0,0],"aroused78A39":[0,1,2,0],"aroused78U07":[2,0,0,0],"aroused79U04":[0,0,0,2],"aroused80A35":[3,0,2,0],"control81E38":[1,0,0,0],"control83E25":[2,1,2,0],"aroused85U32":[1,0,1,3],"aroused87A01":[0,1,0,0],"control87E11":[0,1,0,0],"aroused88A25":[3,0,0,0],"control88E17":[0,0,0,0],"aroused89U23":[1,0,3,1],"aroused91U15":[0,1,1,2],"aroused92A09":[0,1,0,4],"aroused93A33":[1,0,0,0],"control93E20":[1,0,0,0],"control93E34":[0,0,0,0],"aroused94U38":[0,0,0,0],"aroused97A34":[2,3,2,0],"control97E12":[1,0,0,0],"aroused98A20":[0,0,1,1],"control98E13":[1,0,0,0],"control98E28":[0,1,0,1],"aroused99U30":[4,0,3,0]},
"Veroff":{"aroused101": [2,1,1,0,0],"control201": [2,1,0,0,0],"aroused102": [0,0,0,0,0],"control202": [2,0,1,1,0],"aroused103": [3,1,1,2,2],"control203": [0,0,0,0,0],"aroused104": [0,1,0,0,0],"control204": [0,1,0,1,1],"aroused105": [1,1,2,2,0],"control205": [0,0,0,0,0],"aroused106": [0,1,0,1,2],"control206": [2,0,0,0,2],"aroused107": [2,1,0,1,0],"control207": [1,0,1,1,2],"aroused108": [0,3,0,3,2],"control208": [1,2,0,2,3],"aroused109": [1,1,1,2,1],"control209": [0,2,1,0,2],"aroused110": [2,0,1,3,0],"control210": [1,0,0,2,1],"aroused111": [1,1,1,2,0],"control211": [1,0,0,1,0],"aroused112": [0,0,0,2,0],"control212": [0,0,1,0,0],"aroused113": [1,0,0,0,2],"control213": [0,1,1,0,2],"aroused114": [0,2,0,3,2],"control214": [1,1,3,1,0],"aroused115": [0,2,0,2,2],"control215": [0,0,1,1,1],"aroused116": [2,0,1,1,1],"control216": [0,0,0,1,0],"aroused117": [1,1,2,4,0],"control217": [0,0,0,1,0],"aroused118": [0,0,0,0,0],"control218": [1,0,0,1,1],"aroused119": [0,0,0,0,0],"control219": [0,1,0,0,1],"aroused120": [2,0,2,0,1],"control220": [0,1,0,0,0],"aroused121": [0,1,1,2,2],"control221": [0,1,1,0,1],"aroused122": [1,0,0,0,0],"control222": [0,0,0,2,0],"aroused123": [0,0,2,1,0],"control223": [2,1,1,1,1],"aroused124": [0,1,0,2,2],"control224": [1,0,0,1,1],"aroused125": [1,1,0,1,0],"control225": [0,1,0,1,1],"aroused126": [2,1,2,2,0],"control226": [1,1,0,1,1],"aroused127": [2,1,1,1,0],"control227": [1,0,1,3,1],"aroused128": [2,0,1,0,0],"control228": [0,0,1,1,1],"aroused129": [0,2,1,1,1],"control229": [1,1,1,1,2],"aroused130": [2,1,0,0,1],"control230": [1,1,0,0,1],"aroused131": [2,0,0,1,0],"control231": [1,0,0,1,1],"aroused132": [2,2,0,1,2],"control232": [1,1,0,1,1],"aroused133": [0,1,0,0,0],"control233": [2,0,0,0,0],"aroused134": [1,1,0,0,0],"control234": [2,0,0,1,0]},
"Winter":{"aroused101": [3,0,4,1,1,0],"control201": [1,0,3,0,2,1],"aroused102": [2,0,2,2,2,1],"control202": [1,0,0,0,0,0],"aroused103": [2,0,1,1,0,1],"control203": [0,0,1,1,0,0],"aroused104": [3,0,4,1,3,3],"control204": [1,0,0,0,0,0],"aroused105": [0,0,1,1,2,2],"control205": [1,0,1,2,0,0],"aroused106": [1,0,1,0,2,1],"control206": [2,0,3,0,2,0],"aroused107": [2,0,2,1,1,0],"control207": [3,1,1,0,1,0],"aroused108": [2,0,2,0,1,0],"control208": [2,0,0,0,2,1],"aroused109": [2,0,0,1,1,1],"control209": [3,0,3,0,0,0],"aroused110": [3,0,1,1,2,0],"control210": [0,0,0,0,0,0],"aroused111": [3,0,2,0,1,1],"control211": [0,0,2,0,1,0],"aroused112": [0,0,1,1,1,1],"control212": [3,0,1,0,2,0],"aroused113": [1,0,0,1,2,0],"aroused114": [1,0,0,0,2,1],"aroused115": [0,0,1,0,1,0],"aroused116": [1,0,1,2,2,0],"aroused117": [1,0,0,0,0,1],"control01": [1,0,0,0,0,0],"aroused05": [2,0,2,1,1,0],"control06": [2,0,0,0,0,0],"control07": [0,0,2,0,1,0],"control09": [0,0,1,2,2,2],"control11": [2,0,0,0,0,0],"control13": [2,0,1,0,1,2],"aroused14": [3,0,2,2,1,0],"aroused20": [3,0,1,1,0,1],"aroused23": [2,1,3,2,0,2],"control25": [0,0,2,0,0,1],"aroused28": [3,0,3,1,1,3],"control31": [1,0,1,1,1,0],"control36": [1,0,1,0,1,0],"aroused38": [2,0,2,1,0,1],"control43": [4,1,1,1,0,0],"control49": [0,0,2,0,3,1],"control52": [2,0,2,2,1,0],"aroused54": [2,0,2,1,1,2],"aroused55": [0,0,1,1,0,0],"control57": [0,0,1,1,0,0],"aroused60": [4,0,2,2,1,2],"control61": [1,0,0,1,1,0],"aroused69": [1,1,2,1,2,0],"control72": [3,0,2,1,0,1],"aroused74": [0,0,1,1,0,0],"control81": [0,1,1,0,2,0],"control83": [2,0,2,2,0,0],"control89": [2,0,1,0,0,0],"control93": [2,0,3,1,0,0],"aroused97": [1,0,0,0,0,0],"control0129": [0,0,1,1,1,0],"aroused0322": [3,0,0,0,1,0],"aroused0647": [2,0,1,2,1,0],"control0817": [1,0,0,1,1,0],"control0984": [2,2,2,2,0,2],"aroused1026": [3,1,2,3,1,0],"aroused2687": [2,0,1,0,1,0],"control3490": [1,0,1,0,0,0],"aroused3742": [4,0,3,3,2,3],"aroused4214": [1,0,2,3,1,0],"aroused4282": [1,1,2,1,2,1],"control4286": [1,0,1,3,0,0],"control4416": [1,3,2,1,1,1],"aroused4663": [0,0,1,0,1,0],"aroused4916": [0,0,1,2,0,0],"control5073": [2,0,0,0,0,0],"aroused5658": [3,0,2,2,2,1],"control6019": [2,0,2,2,1,0],"control6343": [1,0,0,0,1,0],"aroused6603": [3,0,2,2,1,1],"control7247": [1,0,2,1,2,1],"aroused7277": [3,1,1,1,2,2],"aroused7843": [3,0,2,2,0,3],"control7884": [2,1,2,2,0,0],"aroused7966": [1,0,1,0,0,0],"control8210": [2,1,1,0,2,2],"aroused8518": [2,1,0,2,0,0],"control8868": [2,1,0,3,2,0],"aroused9008": [3,0,1,2,4,0],"aroused9139": [3,1,3,1,1,0],"aroused9717": [2,0,2,0,0,0],}}

if __name__ == "__main__":
	import os, sys

	for dataset in _resultDict.keys():
		result = performCV([dataset])
		print("Best result after optimization of Threshold on {}".format(dataset))
		if os.isatty(sys.stdout.fileno()):
			print("\033[1m" + result.oneline() + "\033[0m")
		else:
			print(result.oneline())
		print(result)
		print("")