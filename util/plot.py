import itertools
import logging
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

