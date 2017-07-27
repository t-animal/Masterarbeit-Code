#!python3
#encoding: utf-8

import json
import os

from collections import defaultdict

from . import isAroused

def getWeightedScore(nestedCVJson):
	outerCVResults = nestedCVJson["nestedCVResult"]["additional"]
	distances = [(path, distance) for result in outerCVResults for path, distance in result["additional"]["distances"].items()]

	percentages = defaultdict(int)
	for path, distance in distances:
		folder, filename = os.path.split(path)
		aroused = isAroused(path)

		imageId = filename[:-5] + ("aroused" if aroused else "control")
		percentages[imageId] += distance

	return percentages


def getJson(path):
	with open(path, "r") as file:
		return json.load(file)

def printScore(weightedScore, verbose = False):
	arousedPos = 0
	arousedNeg = 0
	controlPos = 0
	controlNeg = 0

	sortedScores = sorted(weightedScore.items(), key = lambda x: x[0][-7:] + x[0][:-7])
	for imageId, distance in sortedScores:
		if imageId.endswith("aroused"):
			if distance > 0:
				arousedPos += 1
			if distance < 0:
				arousedNeg += 1
		else:
			if distance > 0:
				controlPos += 1 
			if distance <0:
				controlNeg += 1

		if verbose:
			print("{}:\t{}".format(imageId, distance))

	if verbose:
		print("")
	print("Aroused >0: {} ({:.3f}%)".format(arousedPos, arousedPos/len(weightedScore)*100))
	print("Aroused <0: {} ({:.3f}%)".format(arousedNeg, arousedNeg/len(weightedScore)*100))
	print("Control >0: {} ({:.3f}%)".format(controlPos, controlPos/len(weightedScore)*100))
	print("Control <0: {} ({:.3f}%)".format(controlNeg, controlNeg/len(weightedScore)*100))

	print("Assuming aroused should be >0: {} ({:.3f}%) correct".format(arousedPos+controlNeg, (arousedPos+controlNeg)/len(weightedScore)*100))
	print("Assuming aroused should be <0: {} ({:.3f}%) correct".format(arousedNeg+controlPos, (arousedNeg+controlPos)/len(weightedScore)*100))


def main(args):
	for index, filename in enumerate(args.f):
		result = getJson(filename)
		meta = result["meta"]
		print("Results for {} on {} using {}".format(meta["classifier"], meta["datasets"], os.path.split(meta["optimized"]["modelPath"][0])[1]))
		printScore(getWeightedScore(result))
		print("Unweighted score is: {:.3f} ± {:.3f}".format(result["nestedCVResult"]["correct-percentage-mean"],
		                                                    result["nestedCVResult"]["correct-percentage-stddev"]))

		if not index == len(args.f) - 1:
			print("")

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Sum up distances to the hyperplane and thus give a value per author')
	parser.add_argument("-v",      help = "Be more verbose (repeat v for more verbosity)", action = "count", default = 0)
	parser.add_argument("-f",      help = "Filenames to the nested cv results", action = "store", nargs="+", default = [])

	args = parser.parse_args()
	main(args)