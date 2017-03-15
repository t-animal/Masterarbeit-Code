import numpy as np
import os

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def isAroused(path):
	""" Returns True if path points to a document which was written under aroused conditions
	"""

	folder, filename = os.path.split(os.path.abspath(path))

	if "Veroff" in folder:
		return "E" in filename

	if "Study_2_PSE_files" in folder:
		return int(path[-5]) >= 4

	if "Winter_arousal_stories" in folder:
		return filename.startswith("Winter_1")

	if "AtkinsonEtAl_nAff_cleaned" in folder:
		return "R" in filename

	if "McAdams_1980_nAff_cleaned" in folder:
		return int(filename[:3]) not in ([9, 10, 11, 12] + list(range(85, 123)))

	if "McClelland_et_al_nAch_cleaned" in folder:
		return not "E" in filename

	if "PSE_WirthSchultheiss2006" in folder:
		return "Bridges" in filename

	raise ValueError("Unknown path")