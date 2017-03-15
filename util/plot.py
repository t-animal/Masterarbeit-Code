import logging
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

log = logging.getLogger("de.t_animal.MA.util.pcaPlotter")

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

	plt.matshow(vectors, vmin=-3, vmax=3, norm=matplotlib.colors.SymLogNorm(0.001))
	plt.colorbar()

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
