#!../venv/bin/python

import json
import numpy as np
import scipy.stats
import os
import sys

from collections import defaultdict
from pylatex import *
from pylatex.basic import *
from pylatex.utils import *
from pylatex.package import *
from pylatex.position import *
from pylatex.base_classes import *

class Rotate(ContainerCommand):
	def __init__(self, degrees, data=None, *args, **kwargs):
		super().__init__(*args, data=data, **kwargs)

		self.latex_name = "rotatebox"
		self.arguments = [str(degrees)]
		self.options = None
		self.packages = [Package("graphicx")]


def getAllResults(folder):
	allResults = defaultdict(dict)
	prefixLen = len("testOutput_")
	for fileName in filter(lambda x: x.endswith("json"), os.listdir(folder)):
		key = fileName[prefixLen:fileName.index("_", prefixLen)]
		dataset = fileName[prefixLen + len(key) + 1:-5]

		try:
			with open(os.path.join(folder, fileName)) as file:
				results = json.load(file)

				allResults[key][dataset] = results
		except:
			pass

	return allResults

def patchMissingPValue(allResults):
	""" Fixes old json outputs, where nested CV p-values were not yet aggregated. """
	if "p-value" in list(list(allResults.values())[0].values())[0]["nestedCVResult"]:
		return allResults

	for results in allResults.values(): #models
		for result in results.values(): #datasets
			pValues = [x["p-value"] for x in result["nestedCVResult"]["additional"]]
			fisher = -2 * sum(np.log(pValues))
			combinedPValue = scipy.stats.chi2.sf(fisher, 2 * len(result["nestedCVResult"]["additional"]))
			result["nestedCVResult"]["p-value"] = combinedPValue

	return allResults


def getLatexDoc(title, results):
	def groupSortingKey(key):
		if "." in key:
			mod, retrofit = key.split(".")
			return retrofit + mod
		else:
			return "     " + key

	dataSets = sorted(list(list(results.values())[0].keys()))
	models = sorted(list(results.keys()), key=groupSortingKey)

	bestPercentage = [0] * len(dataSets)
	percentages = []
	stddevs = []
	pValues = []
	for model, results in sorted(allResults.items(), key=lambda x: groupSortingKey(x[0])):
		percentages.append([round(x[1]["nestedCVResult"]["correct-percentage-mean"], 2) for x in sorted(results.items())])
		stddevs.append([round(x[1]["nestedCVResult"]["correct-percentage-stddev"], 2) for x in sorted(results.items())])
		pValues.append([round(x[1]["nestedCVResult"]["p-value"], 4) for x in sorted(results.items())])

	for lineno, line in enumerate(percentages):
		for index, value in enumerate(line):
			if value > bestPercentage[index]:
				bestPercentage[index] = value

	for lineno, line in enumerate(percentages):
		for index, value in enumerate(line):
			if abs(bestPercentage[index] - value) <= 0.05:
				percentages[lineno][index] = bold(percentages[lineno][index])


	geometry_options = {
		"landscape": True,
		"a4paper": True,
		"margin": "0.5in"
	}
	doc = Document("testoutput", geometry_options=geometry_options)
	doc.packages.append(Package("FiraSans", options=["sfdefault"]))

	table = Tabular("l|ccccccc")

	table.add_row([" "] + [Rotate(60, x) for x in dataSets])
	table.add_hline()

	for index, (percentages, stddevs, pValues) in enumerate(zip(percentages, stddevs, pValues)):
		if "." in models[index] and not models[index-1].endswith(models[index][-3:]):
			table.add_empty_row()
		table.add_row([MultiRow(2, data = models[index])] + [NoEscape(str(x) + "\%") for x in percentages])

		table.add_row([""]+[SmallText(NoEscape("$\pm$ {}, p: {}\%".format(s, round(p*100,2) if p < 0.1 else ">10"))) for s, p in zip(stddevs, pValues)])


	# resultsFigure = Figure()
	# resultsFigure.append(LargeText(table))
	# resultsFigure.add_caption(

	doc.append(Section(title))
	centeredTable = Center()
	centeredTable.append(LargeText(table))
	doc.append(centeredTable)
	# doc.append(NewLine())
	doc.append(VerticalSpace("1em"))
	doc.append("Results of  5-fold stratified nested cross validation on various datasets using the given classifier. "
				"P-Values aggregated using Fisher's method")
	doc.generate_pdf(clean_tex=True)


if __name__ == "__main__":
	allResults = patchMissingPValue(getAllResults(sys.argv[1]))
	# print(allResults)

	getLatexDoc(os.path.split(os.path.abspath(sys.argv[1]))[-1], allResults)
