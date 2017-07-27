#!../venv/bin/python
#encoding: utf-8

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

def light(s, *, escape=True):
    if escape:
        s = escape_latex(s)

    return NoEscape(r'\textmd{' + s + '}')

def getAllResults(folder):
	allResults = defaultdict(dict)
	for fileName in filter(lambda x: x.endswith("json"), os.listdir(folder)):
		prefixLen = fileName.index("_") + 1
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
	""" Fixes old json outputs, where nested CV p-values were calculated using fisher's method"""
	for results in allResults.values(): #models
		for result in results.values(): #datasets
			if "p-value" in result["nestedCVResult"]:
				continue
			correctPercentages = [x["correct-percentage"] for x in result["nestedCVResult"]["additional"]]
			combinedPValue = scipy.stats.ttest_1samp(correctPercentages, 50).pvalue
			result["nestedCVResult"]["p-value"] = combinedPValue

	return allResults


def getLatexDoc(title, results):
	def groupSortingKey(key):
		if "." in key:
			mod, retrofit = key.rsplit(".", 1)
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

	geometry_options = {
		"landscape": True,
		"a4paper": True,
		"margin": "0.25in"
	}
	doc = Document("testoutput", geometry_options=geometry_options)
	doc.packages.append(Package("FiraSans", options=["sfdefault"]))
	# doc.packages.append(Package("xcolor", options=["dvipsnames"]))

	table = Tabular("l|c" + "c" *len(dataSets))

	table.add_row([" "] + [Rotate(60, x) for x in dataSets] + ["ø"])
	table.add_hline()

	for index, (percentages, stddevs, pValues) in enumerate(zip(percentages, stddevs, pValues)):
		if "." in models[index] and not models[index-1].endswith(models[index].rsplit(".", 1)[-1]):
			table.add_empty_row()

		displayPercentages = list(percentages)
		for column, value in enumerate(percentages):
			if abs(bestPercentage[column] - value) <= 0.05:
				displayPercentages[column] = bold(percentages[column])

		label = [MultiRow(2, data = models[index])]
		data = [TextColor("lightgray", t) if v > 0.05 else t for t,v in zip([NoEscape(str(x) + "\%") for x in displayPercentages], pValues)]
		table.add_row(label + data + [str(round(sum(percentages)/len(percentages), 2))])

		pText = [("<1\\textperthousand" if p < 0.01 else str(round(p * 100,2))+"\%") if p < 0.1 else ">10\%" for p in pValues]
		pText = [SmallText(NoEscape("$\pm$ {}, p: {}".format(s, p))) for s, p in zip(stddevs, pText)]
		table.add_row([""]+[TextColor("lightgray", t) if v > 0.05 else t for t, v in zip(pText, pValues)] + [""])


	bestPercentage = np.array(bestPercentage)

	doc.append(Section(title))
	centeredTable = Center()
	centeredTable.append(table)
	doc.append(centeredTable)
	doc.append(VerticalSpace("1em"))
	doc.append("Results of two times 5-fold stratified nested cross validation on various datasets using the given classifier.")
	doc.append(NewLine())
	doc.append("Average of best-performing hyperparameters: ")
	doc.append(bold("{:6.3f}% ± {:6.3f}".format(bestPercentage.mean(), bestPercentage.std())))
	doc.generate_pdf(clean_tex=False)


if __name__ == "__main__":
	allResults = patchMissingPValue(getAllResults(sys.argv[1]))
	# print(allResults)

	getLatexDoc(os.path.split(os.path.abspath(sys.argv[1]))[-1], allResults)
