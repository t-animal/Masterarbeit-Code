import numpy as np
import os

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def isAroused(path):
	""" Returns True if path points to a document which was written under aroused conditions
	"""

	folder, filename = os.path.split(os.path.abspath(path))

	if folder.endswith("Veroff"):
		return "C" in path

	if folder.endswith("Study_2_PSE_files"):
		return int(path[-5]) >= 4

	raise ValueError("Unknown path")