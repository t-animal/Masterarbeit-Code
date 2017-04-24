import itertools
import logging
import numpy as np
import matplotlib
import operator

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

log = logging.getLogger("de.t_animal.MA.util.plot")

def plot2D(class1, class2, class1Label, class2Label, reductionFunction=PCA):
	X = class1 + class2
	y =  ["c"] * len(class1) + ["e"] * len(class2)
	mask1 = [True]  * len(class1) + [False] * len(class2)
	mask2 = [False] * len(class1) + [True]  * len(class2)

	pca = reductionFunction(n_components=2)
	X_r = pca.fit(X).transform(X)

	# Percentage of variance explained for each components
	log.info('explained variance ratio (first two components): %s'
	      % str(pca.explained_variance_ratio_))

	fig = plt.figure()

	# print("#X Y Class")
	# for idx, val in enumerate(X_r):
	# 	print("{} {} {}".format(val[0], val[1], (class1Label if mask1[idx] else class2Label).replace(" ", "_")))

	plt.scatter(X_r[mask1, 0], X_r[mask1, 1], c="r", label=class1Label, marker="o")
	plt.scatter(X_r[mask2, 0], X_r[mask2, 1], c="g", label=class2Label, marker="^")

	plt.show()

def plot3D(class1, class2, class1Label, class2Label, reductionFunction=PCA):
	X = class1 + class2
	y =  ["c"] * len(class1) + ["e"] * len(class2)
	mask1 = [True]  * len(class1) + [False] * len(class2)
	mask2 = [False] * len(class1) + [True]  * len(class2)

	pca = reductionFunction(n_components=3)
	X_r = pca.fit(X).transform(X)

	# Percentage of variance explained for each components
	log.info('explained variance ratio (first three components): %s'
	          % str(pca.explained_variance_ratio_))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(X_r[mask1, 0], X_r[mask1, 1], X_r[mask1, 2], c="r", label=class1Label, marker="o")
	ax.scatter(X_r[mask2, 0], X_r[mask2, 1], X_r[mask2, 2], c="g", label=class2Label, marker="^")

	plt.show()


def plotHistogram(class1, class2=None):
	fig, ax = plt.subplots()
	bins = sorted(map(lambda x: -x/1000, range(100, 5000, 100)) + map(lambda x: x/1000, range(0, 5000, 100)))
	ax.hist(class1, bins=bins, color="r"*len(class1))
	if class2:
		ax.hist(class2, bins=bins, color="g"*len(class2))
	plt.show()


def plotVectorList(class1, class2=None):
	if class2:
		class1 = class1 + [np.array([-99]*class1[0].shape[0])]*2 + class2
	vectors = np.transpose(np.array(class1))

	cmap = matplotlib.cm.get_cmap("viridis")
	cmap.set_over("r")
	cmap.set_under("r")

	fig, ax = plt.subplots(figsize=(15, 15))
	img = ax.matshow(vectors, vmin=-3, vmax=3, norm=matplotlib.colors.SymLogNorm(0.001))
	plt.colorbar(img)

	plt.xlabel("Texts in Dataset")
	plt.ylabel("Entries in vector")

	# plt.savefig("../vortraege/antrittsvortrag/images/Veroff_pow2_glove_wns+.png", dpi=600)
	plt.show()


def plotPCA(class1, class2, class1Label, class2Label, nDims=2):
	if nDims == 2:
		plot2D(class1, class2, class1Label, class2Label, PCA)
	elif nDims == 3:
		plot3D(class1, class2, class1Label, class2Label, PCA)
	else:
		log.error("nDims must be either 2 or 3, is %s", nDims)

def plotLDA(class1, class2, class1Label, class2Label, nDims=2):
	if nDims == 2:
		plot2D(class1, class2, class1Label, class2Label, LinearDiscriminantAnalysis)
	elif nDims == 3:
		plot3D(class1, class2, class1Label, class2Label, LinearDiscriminantAnalysis)
	else:
		log.error("nDims must be either 2 or 3, is %s", nDims)



def plotPairDistances(class1, class2):
	"""Takes the distance of all vector 2-tuples per class, drops vectors of duplicates and
	   plots the histogram."""
	tuples1 = itertools.filterfalse(lambda x: np.array_equal(*x), itertools.product(class1, class1))
	tuples2 = itertools.filterfalse(lambda x: np.array_equal(*x), itertools.product(class2, class2))

	print(class1[0][0])

	distances1 = list(map(lambda x: np.linalg.norm(x[0] - x[1]), tuples1))
	distances2 = list(map(lambda x: np.linalg.norm(x[0] - x[1]), tuples2))

	maxDistance = max(max(distances1), max(distances2))

	distances1 = [x/maxDistance for x in distances1]
	distances2 = [x/maxDistance for x in distances2]

	bins = [n/500 for n in range(0, 500)]

	fig, ax = plt.subplots()
	ax.hist(distances1, bins = bins, edgecolor="r", histtype = 'step')
	ax.hist(distances2, bins = bins, edgecolor="g", histtype = 'step')
	plt.show()



def plotCrossValResults(results, hyperparam1, hyperparam2, fixedParams = {}, hyperparam1Label = None, hyperparam2Label = None, figureName = None):
	"""Plots the results of a crossvalidation of two hyperparameters as a 2d contour plot
	   (like in A Practical Guide to Support Vector Classification). If the crossvalidation was performed on more than
	   two hyperparameters, add ALL other hyperparameters to the fixedParams dictionary. In this dictionary keys are
	   hyperparamters and values are hyperparameter values. The plot will exclude non-matching fixed hyperparameters. I.e.
	   if you have optimized hyperparameters (X,Y,Z) each in the range(0,4) and want to plot X and Y, choose Z to a fixed
	   value (e.g. 1) and pass it in fixedParams as {"Z": 1}. Params X and Y will be plotted only for those values where
	   Z is 1.

	   :param results: the results to plot as returned by CrossValidation.getCVResults or loaded from a json file generated by crossVal.py
	   :param hyperparam1: which hyperparameter to plot on the x-axis
	   :param hyperparam2: which hyperparameter to plot on the y-axis
	   :param fixedParams: filter for there additional hyperparameter values
	   :param hyperparam1Label: which label to use for the x-axis (defaults to the hyperparamter)
	   :param hyperparam2Label: which label to use for the y-axis (defaults to the hyperparamter)
	   :param figureName: the name of the figure. useful if more than one result is plotted. if provided, the call will be non blocking
	   :type results: list of tuples (hyperparameters (dict(str=>val) or str), util.CrossValidationResultContainer)
	   :type hyperparam1: string
	   :type hyperparam2: string
	   :type fixedParams: dict (str => value)
	   :type hyperparam1Label: string
	   :type hyperparam2Label: string
	   :type figureName: string
	"""

	plt.figure(figureName)

	if type(next(iter(results.keys()))) is str:
		#the results were loaded from a JSON file, restore hyperparam dict
		results = [(dict(eval(k.replace(":", ","))), v) for k,v in results.items()]

	unfixedParams = set(results[0][0].keys()) - set(fixedParams.keys()) - set([hyperparam1, hyperparam2])
	if len(unfixedParams) > 0:
		if unfixedParams == set(["modelPath"]):
			log.warn("The hyperparameter modelPath was not fixed, if one output file per model was created, that's ok")
		else:
			unfixedParams -= set(["modelPath"])
			log.warn("These hyperparameters have not been fixed and might scew the displayed results: {}".format(unfixedParams))

	try:
		filteredResults = OrderedDict([((params[hyperparam1], params[hyperparam2]), result["correct-percentage-mean"]) for params, result in results \
			                                if all([str(params[fixedParam]) == str(fixedValue) for fixedParam, fixedValue in fixedParams.items()])])
		filteredResults = sorted(filteredResults.items(), key=operator.itemgetter(0)) #sort parameters by hyperparam1 first, then hyperparam2
	except KeyError:
		raise ValueError("Invalid hyperparameters provided")

	X = sorted(set([params[0] for params, _ in filteredResults]))
	Y = sorted(set([params[1] for params, _ in filteredResults]))
	Z = np.array([result for _, result in filteredResults]).reshape(len(X), len(Y))

	if len(Z) == 0:
		raise ValueError("No results for these hyperparameters")

	#check if Z contains only equal values
	if(all((Z/Z[0][0] == np.ones((4,4))).reshape(1,-1)[0])):
		raise ValueError("These hyperparameters all produced the value {}. Cannot plot contours".format(Z[0][0]))

	label1 = hyperparam1 if hyperparam1Label is None else hyperparam1Label
	label2 = hyperparam2 if hyperparam2Label is None else hyperparam2Label

	plt.contour(X, Y, Z)
	plt.xlabel(label1)
	plt.ylabel(label2)

	plt.show(block = figureName is None)



if __name__ == "__main__":
	import argparse, json, os
	from util import ArgSplit

	parser = argparse.ArgumentParser(description='Module for plotting. Some functions are exposed to the CLI.')
	parser.add_argument("-v",      help = "Be more verbose (repeat v for more verbosity)", action = "count", default = 0)

	parser.add_argument("--json",  "-j", help = "JSON output of a crossvalidation run",
	                                     type = argparse.FileType('r', encoding='UTF-8'), nargs = "+", default = [])
	parser.add_argument("--hyperparams", "-p", help = "The two hyperparameters to plot", nargs = 2, required = True)
	parser.add_argument("--fixed", "-f", help = "Additional hyperparameters which should be fixed to a certain value", action=ArgSplit, nargs="+", default={})

	args = parser.parse_args()

	for file in args.json:
		try:
			content = json.load(file)
			plotCrossValResults(content, args.hyperparams[0], args.hyperparams[1], args.fixed, figureName = os.path.basename(file.name))
		except Exception as e:
			log.error(e)

	try:
		plt.show()
	except KeyboardInterrupt:
		pass
