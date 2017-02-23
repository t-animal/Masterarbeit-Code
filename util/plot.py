import logging
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

log = logging.getLogger("de.t_animal.MA.util.pcaPlotter")

def plotPCA2D(class1, class2, class1Label, class2Label):
	X = class1 + class2
	y =  ["c"] * len(class1) + ["e"] * len(class2)
	mask1 = [True]  * len(class1) + [False] * len(class2)
	mask2 = [False] * len(class1) + [True]  * len(class2)

	pca = PCA(n_components=2)
	X_r = pca.fit(X).transform(X)

	# Percentage of variance explained for each components
	log.info('explained variance ratio (first two components): %s'
	      % str(pca.explained_variance_ratio_))

	fig = plt.figure()

	plt.scatter(X_r[mask1, 0], X_r[mask1, 1], c="r", label=class1Label, marker="o")
	plt.scatter(X_r[mask2, 0], X_r[mask2, 1], c="g", label=class2Label, marker="^")

	plt.show()

def plotPCA3D(class1, class2, class1Label, class2Label):
	X = class1 + class2
	y =  ["c"] * len(class1) + ["e"] * len(class2)
	mask1 = [True]  * len(class1) + [False] * len(class2)
	mask2 = [False] * len(class1) + [True]  * len(class2)

	pca = PCA(n_components=3)
	X_r = pca.fit(X).transform(X)

	# Percentage of variance explained for each components
	log.info('explained variance ratio (first three components): %s'
	          % str(pca.explained_variance_ratio_))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(X_r[mask1, 0], X_r[mask1, 1], X_r[mask1, 2], c="r", label=class1Label, marker="o")
	ax.scatter(X_r[mask2, 0], X_r[mask2, 1], X_r[mask2, 2], c="g", label=class2Label, marker="^")

	plt.show()


def plotPCA(class1, class2, class1Label, class2Label, nDims=2):
	if nDims == 2:
		plotPCA2D(class1, class2, class1Label, class2Label)
	elif nDims == 3:
		plotPCA3D(class1, class2, class1Label, class2Label)
	else:
		log.error("nDims must be either 2 or 3, is %s", nDims)