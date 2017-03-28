import numpy as np
import os
import socket

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
