import argparse
import numpy as np
import os
import socket
import warnings


#This table is taken from the .docx included with the study2 files
#The first column is the test person ID
#The second indicates image order (two sets of images were shown,
#one before and one after arousal via a movie. One group received 3 images (a,b,c) before the movie and 3 other images
# after the movie (d,e,f), the other group received (d,e,f) before and (a,b,c) after the movie
#The third column indicates test-group (no arousing movie shown = 1, arousing affiliation = 2
#arousing power = 3 )
#unfortunately we don't have information for all stories and all person IDs
_study2_table = [[1124,	1,	1],
				[1133,	0,	1],
				[1224,	1,	2],
				[1233,	0,	2],
				[1324,	1,	3],
				[1333,	0,	3],
				[2121,	1,	1],
				[2125,	1,	1],
				[2134,	0,	1],
				[2221,	1,	2],
				[2225,	1,	2],
				[2234,	0,	2],
				[2321,	1,	3],
				[2325,	1,	3],
				[2334,	0,	3],
				[3122,	1,	1],
				[3131,	0,	1],
				[3135,	0,	1],
				[3222,	1,	2],
				[3231,	0,	2],
				[3235,	0,	2],
				[3322,	1,	3],
				[3331,	0,	3],
				[3335,	0,	3],
				[4123,	1,	1],
				[4132,	0,	1],
				[4223,	1,	2],
				[4232,	0,	2],
				[4323,	1,	3],
				[4332,	0,	3]]
_study2_index = { ID: {"order": order, "aroused": aroused > 1} for ID, order, aroused in _study2_table}

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
		warnings.warn("Study2 has huge problems concerning arousal conditions and image ID. Handle with care.")
		return int(path[-5]) >= 4# and _study2_index[int(filename[2:6])]["aroused"]

	if "Winter_arousal_stories" in folder:
		return filename.startswith("Winter_1")

	if "AtkinsonEtAl_nAff_cleaned" in folder:
		return "R" in filename

	if "McAdams_1980_nAff_cleaned" in folder:
		return int(filename[:3]) not in ([9, 10, 11, 12] + list(range(85, 123)))

	if "McClelland_et_al_nAch_cleaned" in folder:
		return not "E" in filename

	if "PSE_WirthSchultheiss2006" in folder:
		return "Bridges" in filename and int(filename[-5]) >= 4

	raise ValueError("Unknown path")


def getImageID(path):
	""" Return for which image a story was written. Except for Study2 and WirthSchultheiss
	    these numbers are speculative, though! We do have to assume story order was not randomized"""

	folder, filename = os.path.split(os.path.abspath(path))

	if "Veroff" in folder:
		warnings.warn("Using image ID for the Veroff dataset is speculative!")
		return int(filename[-5])

	if "Study_2_PSE_files" in folder:
		#see comment above for _study2_table
		warnings.warn("Study2 has huge problems concerning arousal conditions and image ID. Handle with care.")
		return 1
		apparentId = int(filename[-5])
		if _study2_index[int(filename[2:6])]["order"] == 1:
			return apparentId
		else:
			return (apparentId + 2) % 6 + 1

	if "Winter_arousal_stories" in folder:
		warnings.warn("Using image ID for the Winter dataset is speculative!")
		return int(filename[-5])

	if "AtkinsonEtAl_nAff_cleaned" in folder:
		warnings.warn("Using image ID for the Atkinson dataset is speculative!")
		return int(filename[-5])

	if "McAdams_1980_nAff_cleaned" in folder:
		warnings.warn("Using image ID for the McAdams dataset is speculative!")
		return int(filename[-5])

	if "McClelland_et_al_nAch_cleaned" in folder:
		warnings.warn("Using image ID for the McClelland dataset is speculative!")
		return int(filename[-5])

	if "PSE_WirthSchultheiss2006" in folder:
		warnings.warn("Story order was randomized in Wirth dataset, we have no information on image ID")
		return 1

	raise ValueError("Unknown path")


def set_keepalive_linux(sock, after_idle_sec=5, interval_sec=15, max_fails=5):
    """Set TCP keepalive on an open socket.

    It activates after 1 second (after_idle_sec) of idleness,
    then sends a keepalive ping once every 3 seconds (interval_sec),
    and closes the connection after 5 failed ping (max_fails), or 15 seconds
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, after_idle_sec)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, max_fails)

class ArgSplit(argparse.Action):
	def __init__(self, option_strings, dest, **kwargs):
		self.dest = dest
		super(ArgSplit, self).__init__(option_strings, dest, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):

		if any(["=" not in v for v in values]):
			raise ValueError("Args must be key value pairs, joined by a = (key=value)")

		setattr(namespace, self.dest, dict([val.split("=", 1) for val in values]))