#!../venv/bin/python

import json
import os
import sys

from collections import defaultdict
from pylatex import Document, Section, Subsection, Tabular, MultiColumn,\
    MultiRow
from pylatex.basic import *
from pylatex.utils import *
from pylatex.package import *

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

def getLatexDoc(results):
	def groupSortingKey(key):
		if "." in key:
			mod, retrofit = key.split(".")
			return retrofit + mod
		else:
			return "     " + key

	dataSets = sorted(list(list(results.values())[0].keys()))
	models = sorted(list(results.keys()), key=groupSortingKey)

	bestLine = [0] * len(dataSets)
	percentages = []
	stddevs = []
	for model, results in sorted(allResults.items(), key=lambda x: groupSortingKey(x[0])):
		percentages.append([round(x[1]["nestedCVResult"]["correct-percentage-mean"], 2) for x in sorted(results.items())])
		stddevs.append([round(x[1]["nestedCVResult"]["correct-percentage-stddev"], 2) for x in sorted(results.items())])

	for lineno, line in enumerate(percentages):
		for index, value in enumerate(line):
			if value > percentages[bestLine[index]][index]:
				bestLine[index] = lineno

	for index, lineno in enumerate(bestLine):
		percentages[lineno][index] = bold(percentages[lineno][index])


	geometry_options = {
		"landscape": True,
		"a4paper": True
	}
	doc = Document("testoutput", geometry_options=geometry_options)
	doc.packages.append(Package("FiraSans", options=["sfdefault"]))
	table = Tabular("l|ccccccc")

	table.add_row([" "] + dataSets)
	table.add_hline()

	for index, (percentages, stddevs) in enumerate(zip(percentages, stddevs)):
		if "." in models[index] and not models[index-1].endswith(models[index][-3:]):
			table.add_empty_row()
		table.add_row([MultiRow(2, data = models[index])] + [NoEscape(str(x) + "\%") for x in percentages])
		table.add_row([""]+[NoEscape("$\pm$ " + str(x)) for x in stddevs])


	doc.append(LargeText(table))
	doc.generate_pdf(clean_tex=True)


if __name__ == "__main__":
	allResults = getAllResults(sys.argv[1])

	getLatexDoc(allResults)
