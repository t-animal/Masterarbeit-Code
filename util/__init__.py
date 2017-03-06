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
		return "E" in filename

	if folder.endswith("Study_2_PSE_files"):
		return int(path[-5]) >= 4

	if folder.endswith("Winter_arousal_stories"):
		return filename.startswith("Winter_1")

	if folder.endswith("AtkinsonEtAl_nAff_cleaned"):
		return "R" in filename

	if folder.endswith("McAdams_1980_nAff_cleaned"):
		return int(filename[:3]) not in ([9, 10, 11, 12] + list(range(85, 123)))

	raise ValueError("Unknown path")