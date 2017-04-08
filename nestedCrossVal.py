#!./venv/bin/python
# PYTHON_ARGCOMPLETE_OK

from collections import OrderedDict
from tester import getClassifierClass
from util import set_keepalive_linux
from util.containers import CrossValidationResultContainer
from util.argGenerator import generateCrossValidationSets

import binascii
import hashlib
import itertools
import json
import logging
import multiprocessing
import pickle
import random
import socket
import socketserver
import string
import threading

log = logging.getLogger("de.t_animal.MA.nestedCrossVal")

class _signals():
	OK = b'\x00'
	ABORT = b'\x01'

#This should be around 500000
N_ROUNDS=500


class NestedCV:

	def getNestedCVResult(self, classifierName, crossValidateSet, optimize):
		""" Gets the nested cross validation result


	    :param classifierName: the name of the classifier to use
	    :param crossSet: the name of the data set to crossvalidate on
	    :param optimize: a dict where each key is a hyperparameter and each value is a list of possible candidates
	    :type classifierName: string
	    :type crossSet: string
	    :type optimize: dict
		"""
		raise NotImplemented()


	@staticmethod
	def getInnerCVResults(classifierName, crossSet, classifierArgs, includeOuterResults = False):
		""" Performs a nested cross validation and returns a list of the results of the inner cross validations.
		    If includeOuterResults is True, the list contains tuples of the result of the outer validation and
		    the inner cross validation.

		    :param classifierName: the name of the classifier to use
		    :param crossSet: the name of the data set to crossvalidate on
		    :param classifierArgs: the kw-parameters to pass to the classifier
		    :param includeOuterResults: whether to include the outer results or not
		    :type classifierName: string
		    :type crossSet: list of strings
		    :type classifierArgs: dict
		    :type includeOuterResults: boolean

		    :returns: list of CrossValidateResultContainer or
		              list of tuple (TestResultContainer, CrossValidateResultContiner)
		    """

		testResults = []
		classifier = getClassifierClass(classifierName)(**classifierArgs)

		for crossTestSet in generateCrossValidationSets(crossSet):
			innerTestResult = CrossValidationResultContainer("aroused", "nonAroused")

			for crossValidateSet in crossTestSet["crossValidate"]:
				classifier.train(crossValidateSet["train"])
				innerTestResult.addResult(classifier.test(crossValidateSet["validate"]))

			outerTrainSet = crossTestSet["crossValidate"][0]["train"] + crossTestSet["crossValidate"][0]["validate"]
			classifier.train(outerTrainSet)
			outerTestResult = classifier.test(crossTestSet["outerValidate"])

			if includeOuterResults:
				testResults.append((outerTestResult, innerTestResult))
			else:
				testResult.append(innerTestResult)

		log.info("Finished a computation.")
		return testResults

	@staticmethod
	def getKWTuples(p_grid):
		""" Returns all combinations of all keys and entries of lists that are the values of a dict.
		I.e. {1: [2,3], 4: [5,6]} => [{1:2, 4:5}, {1:3, 4:5}, {1:2, 4:6}, {1:3, 4:6}].
		"""
		tuples = list(itertools.product(*p_grid.values()))
		kwTuples = []

		for simpleTuple in tuples:
			kwTuples.append({key:simpleTuple[index] for index, key in enumerate(p_grid.keys())})

		return kwTuples


class ParallelNestedCV(NestedCV):

	def __init__(self, n_jobs = 4):
		self.pool = multiprocessing.Pool(n_jobs)

	def getNestedCVResult(self, classifierName, crossValidateSet, optimize):
		log.info("Beginning optimization.")
		classifierArgsList = self.getKWTuples(optimize)
		testResultList = self.pool.map(ParallelNestedCV._getInnerCVResultsMapProxy, [(classifierName, crossValidateSet, arg) for arg in classifierArgsList])

		return self.getBestResultsFromParallelComputation(testResultList)

	@staticmethod
	def getBestResultsFromParallelComputation(testResultList):
		"""Similar to the reduce-step of Map-Reduce this method takes a list of results from
		_getInnerCVResultsMapProxy, finds the best argument tuple of the inner CVs for each step
		of the outer CV. It then returns the result of the outer CV based on the pre-computed outerCV
		results for each of the best argument tuples. It also returns a list of tuples of best-
		performing arguments and their results per inner CV.

		:param testResultList: mapped results of _getInnerCVResultsMapProxy
		:type testResultList: list of tuple (dict, tuple (TestresultContainer, CrossValidationResultContainer))

		:returns: tuple (CrossValidationResultContainer, list of tuple(dict, CrossValidationResultcontainer))
		"""
		bestResultPerInnerCV = []
		resultOuterCVPerBestInnerCV = []
		argsPerBestInnerCV = []
		for args, cvResults in testResultList:
			for index, (outerTestResult, innerTestResult) in enumerate(cvResults):
				try:
					if bestResultPerInnerCV[index] < innerTestResult:
						bestResultPerInnerCV[index] = innerTestResult
						resultOuterCVPerBestInnerCV[index] = outerTestResult
						argsPerBestInnerCV[index] = args
				except IndexError:
					bestResultPerInnerCV.append(innerTestResult)
					resultOuterCVPerBestInnerCV.append(outerTestResult)
					argsPerBestInnerCV.append(args)

		crossValResult = CrossValidationResultContainer("aroused", "nonAroused")
		for result in resultOuterCVPerBestInnerCV:
			crossValResult.addResult(result)

		return crossValResult, zip(argsPerBestInnerCV, bestResultPerInnerCV)


	@staticmethod
	def _getInnerCVResultsMapProxy(args):
		"""Helper method for mapping getInnerCVResults. It returns a tuple of
		   the passed parameters KW tuples (i.e. the parameters passed to the classifier) and
		   the results of getInnerCVResults.

			:param args: a tuple consisting of the arguments passed to getInnerCVResults
			:type args: tuple (string,  list of strings, dict)

			:returns: tuple (dict, tuple (TestresultContainer, CrossValidationResultContainer))
		"""
		return (args[2], NestedCV.getInnerCVResults(*args, includeOuterResults = True))


class ExecutionRequestHandler(socketserver.BaseRequestHandler):
	def handle(self):
		"""Handles an execution request passed on by a ExecutionRequestServer (aka worker).

		Communication with the master is as follows:
			1) Worker sends a random, 64byte string
			2) Server sends the pbkdf_hmac of the shared secret using the random string as salt
			3) Worker sends the _signals.OK or closes connection
			4) Server sends a 4-byte integer I in Big Endian indicating the length of the following information
			5) Server sends the pickled computation request of size I (in bytes)
			6) Worker passes the computation request to ParallelNestedCV._getInnerCVResultsMapProxy
			7) When worker receives the _signals.ABORT before 8) it stops all computation on this server (including other workers!)
			8) When the computation request is finished the worker sends a 4-byte integer J in Big Endian indicating
			   the length of the following information
			9) The Worker sends the computation result of length J
		"""
		poolEpoch = self.server.poolEpoch

		#basic authentication, no need for higher security atm
		salt = "".join([random.choice(string.hexdigits) for x in range(0, 64)])
		authcode = hashlib.pbkdf2_hmac("sha512", self.server.secret.encode("ascii"),
		                               salt.encode("ascii"), N_ROUNDS)

		self.request.sendall(salt.encode("ascii"))

		if not binascii.hexlify(authcode) == self.request.recv(129):
			log.warning("Requester could not authenticate")
			return

		self.request.sendall(_signals.OK)

		#authenticated, loading arguments
		size = int.from_bytes(self.request.recv(4), byteorder = "big")
		pickled = self.request.recv(1024)
		while len(pickled) < size:
			pickled += self.request.recv(1024)
		args = pickle.loads(pickled)

		log.info("Received computation request: {}({}) on {}".format(args[0], args[2], args[1]))

		result = self.server.pool.apply_async(ParallelNestedCV._getInnerCVResultsMapProxy, (args,))

		self.request.settimeout(5.0)
		while True:
			try:
				if self.request.recv(1) == _signals.ABORT:
					log.info("Received a termination request.")
					self.server.terminatePool(poolEpoch)
					return
			except socket.timeout:
				pass

			if result.ready():
				break

		try:
			result = {"result": result.get(), "exception": None}
		except Exception as e:
			result = {"result": None, "exception": e}

		self.request.settimeout(None)
		answer = pickle.dumps(result, protocol=3)
		self.request.send(len(answer).to_bytes(4, byteorder="big"))
		self.request.sendall(answer)


class ExecutionRequestServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
	""" Simple TCPServer class accepting execution requests from masters and passing them on
	    to ExecutionRequestHandlers"""

	allow_reuse_address = True

	def __init__(self, port, secret, n_cpus):
		""" Usually not needed, use startWorker to start a worker and let it run until killed.

		:param port: the port to listen on
		:param secret: the secret to secure the connection to masters with
		:param n_cpus: the maximum amount of processes to start
		:type port: integer
		:type secret: string
		:type n_cpus: integer > 0
		"""
		super().__init__(("0.0.0.0", port), ExecutionRequestHandler)

		self.secret = secret
		self.n_cpus = n_cpus
		self.pool = multiprocessing.Pool(n_cpus)
		self.poolEpoch = 0
		self.terminationLock = threading.Lock()

	def terminatePool(self, poolEpoch):
		""" Terminates the pool of workers. This stops all computation still running on this server!
		    Called only from the ExecutionRequestHandler

		:param poolEpoch: the current epoch from the handler's view. ensures only the first handler's termination request is executed
		:type poolEpoch: integer
		"""
		with self.terminationLock:
			if not poolEpoch == self.poolEpoch:
				return

			self.pool.terminate()
			self.pool.join()
			self.pool = multiprocessing.Pool(self.n_cpus)
			self.poolEpoch += 1

	@staticmethod
	def startWorker(secret, port, n_cpus):
		""" Starts a worker and serves forever. Does not return unless an Exception is raised!

		:param secret: the secret to secure the connection to masters with
		:param port: the port to listen on
		:param n_cpus: the maximum amount of processes to start
		:type secret: string
		:type port: integer
		:type n_cpus: integer > 0
		"""
		server = ExecutionRequestServer(port, secret, n_cpus)
		server.serve_forever()


class ExecutionRequestMaster(ParallelNestedCV):

	def __init__(self, secret, workersPort, workers):
		self.secret = secret
		self.workersPort = workersPort

		self.workers = workers
		self.loopedWorkers = itertools.cycle(workers)

		self.workerSocks = []
		self.failedWorkers = []
		self.assignedTasks = {}

	def getWorkerSock(self, worker):
		"""Connects and authenticates to a worker

		:param worker: the worker to connect to (hostname, fqdn, ip...)
		:type worker: string"""
		try:
			sock = socket.create_connection((worker, self.workersPort), 5)
			set_keepalive_linux(sock) # not platform independant, could be deleted if your NAT is not as shitty as mine
		except Exception as e:
			log.debug(e)
			return None

		salt = sock.recv(64)
		authcode = hashlib.pbkdf2_hmac("sha512", self.secret.encode("ascii"), salt, N_ROUNDS)

		sock.sendall(binascii.hexlify(authcode))

		if not sock.recv(1) == _signals.OK:
			log.warning("Could not authenticate on worker %s", worker)
			return None

		#authenticated, adding to list
		return sock

	def dispatchWork(self, classifierName, classifierArgs, crossValidateSets):
		"""Dispatches a work assignment to the next worker.

	    :param classifierName: the name of the classifier to use
	    :param classifierArgs: the dict of kw-parameters to pass to the classifier
	    :param crossValidateSets: the data sets to work upon
	    :type classifierName: string
	    :type classifierArgs: dict
	    :type crossValidateSets: list of strings
	    """
		worker = next(self.loopedWorkers)
		workerSock = self.getWorkerSock(worker)

		while workerSock is None:
			log.warn("Worker %s failed to connect or authenticate", worker)
			self.failedWorkers.append(worker)
			if set(self.workers) == set(self.failedWorkers):
				log.error("All workers have failed!")
				raise KeyboardInterrupt() # maybe some workers have worked before, clean them up

			worker = next(self.loopedWorkers)
			workerSock = self.getWorkerSock(worker)

		self.workerSocks.append((worker, workerSock))

		#assumptions here are: either this fails on all workers or none; all workers have roughly the same speed
		data = pickle.dumps((classifierName, crossValidateSets, classifierArgs), protocol=3)
		workerSock.send(len(data).to_bytes(4, byteorder="big"))
		workerSock.sendall(data)

		self.assignedTasks[workerSock] = classifierArgs


	def getNestedCVResult(self, classifierName, crossValidateSets, optimize):
		kwargsList = self.getKWTuples(optimize)

		try:
			log.info("Beginn dispatching work assignments")
			for classifierArgs in kwargsList:
				self.dispatchWork(classifierName, classifierArgs, crossValidateSets)
			log.info("Dispatched all work assignments")

			while len(self.assignedTasks) > 0:
				testResultList = []
				for index, (worker, workerSock) in enumerate(self.workerSocks):
					try:
						size = int.from_bytes(workerSock.recv(4), byteorder="big")
						pickled = workerSock.recv(1024)
						while len(pickled) < size:
							pickled += workerSock.recv(1024)
						result = pickle.loads(pickled)
					except KeyboardInterrupt:
						raise KeyboardInterrupt()
					except socket.timeout:
						continue
					except Exception as exception:
						log.warning("Could not get results from worker %s due to exception: %s. Redispatching", worker, exception)
						self.dispatchWork(classifierName, self.assignedTasks[workerSock], crossValidateSets)
						continue
					finally:
						del self.assignedTasks[workerSock]

					if result["exception"] is not None:
						log.warning("Worker %s threw an exception: %s. NOT REDISPATCHING!", worker, result["exception"])
					else:
						log.info("Worker %s has finished a computation (%d/%d done)", worker, len(testResultList) + 1, len(kwargsList))
						testResultList.append(result["result"])

		except KeyboardInterrupt:
			log.error("Exiting, sending workers kill signal.")
			log.error("This terminates all computation on all connected workers.")
			log.error("Give the workers a couple of seconds to reinitiate.")
			try:
				for worker, workerSock in self.workerSocks:
					workerSock.send(_signals.ABORT)
			except BrokenPipeError:
				pass

			return None, None

		return self.getBestResultsFromParallelComputation(testResultList)




if __name__ == "__main__":

	import argparse,configparser, os, sys, json

	configParser = configparser.ConfigParser()
	configParser.read(os.path.split(__file__)[0] + os.sep + "tester.ini")
	if "ModelPaths" in configParser:
		modelPaths = dict(configParser["ModelPaths"])
	else:
		modelPaths = {}

	from tester import classifierCompleter, ChoicesContainer, _dataSets

	parser = argparse.ArgumentParser(description='Optimize for hyperparameters (on a cluster). Requires a file secretFile to be readable.')
	parser.add_argument("--json",  help = "Display the output as json",              action = "store_true")
	parser.add_argument("-v",      help = "Be more verbose (-vv for max verbosity)", action = "count", default = 0)

	parser.add_argument("--secretFile",  "-s", help = "A file containing a shared secret for all workers and the master (e.g. `dd if=/dev/urandom bs=1 count=100 | base64 > secretFile`)",
							type = argparse.FileType('r', encoding='UTF-8'), default = "secretFile")

	# parser.add_argument("classifierArgs",     help = "additional arguments to pass to the classifier (overrides optimized parameters!)",
	#                                nargs = "*")

	commandParsers = parser.add_subparsers(title="action", help="Whether to start a worker or a master node", dest="action")

	workerParser = commandParsers.add_parser("worker", help="Start a worker node")
	workerParser.add_argument("--port", "-p",  help = "The port to listen on (default: 23654)", type = int, default = 23654)
	workerParser.add_argument("--num",  "-n", help = "The maximum cpus to use for worker processes (default: all)", type = int, default = multiprocessing.cpu_count())


	masterParser = commandParsers.add_parser("master", help="Start a master node and optimize on the given worker nodes")
	masterParser.add_argument("--port", "-p",  help = "The port to connect to on the workers (default: 23654)", type = int, default = 23654)

	masterParser.add_argument("--classifier", "-c", help = "Which classifier to use", required = True) \
					   .completer = classifierCompleter
	masterParser.add_argument("--model", "-m",  help = "Load the model path from 'tester.ini' and pass it to the classifier",
								   choices = list(modelPaths.keys()))
	masterParser.add_argument("--datasets", "-d", help = "The datasets to crossvalidate on (must be available on all cluster workers)",
								   nargs = "+", choices = ChoicesContainer(_dataSets.keys()), required = True)
	masterParser.add_argument("-o", "--outFile", help = "Write json output to this file, too (will be truncated)", type = argparse.FileType("w"))
	masterParser.add_argument("optimize",  help = "The options to optimize as a dict, passed as json (e.g. '{\"C\":[1,2,3], \"foo\":[4.6,5.7]}'", type = json.loads, default = None)


	localParser = commandParsers.add_parser("local", help="like master, but don't connect to any workers, optimize on this machine only",
												parents=[masterParser], conflict_handler='resolve')

	masterParser.add_argument("--workers", "-w",  help = "The clients' hostnames to connect to", type = str, nargs = "+", required = True)

	args = parser.parse_args()


	if args.action == "worker" or args.action == "local":
		logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level = logging.WARN)
		#if loglevel is 1 print only worker logs, if greater print lower level logs, too
		if args.v == 1:
			log.setLevel(logging.INFO)
		elif args.v > 1:
			logging.basicConfig(level=[logging.WARN, logging.INFO, logging.DEBUG][min(args.v - 1, 2)])
	else:
		logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		                level=[logging.WARN, logging.INFO, logging.DEBUG][min(args.v, 2)])

	if args.action == "worker":
		ExecutionRequestServer.startWorker(args.secretFile.read(), args.port, args.num)
		args.secretFile.close()
		sys.exit(0)

	if args.model:
		args.optimize["modelPath"] = [modelPaths[args.model]]

	if args.action == "master":
		master = ExecutionRequestMaster(args.secretFile.read(), args.port, args.workers)
		bestItem, innerCVResults = master.getNestedCVResult(args.classifier, args.datasets, args.optimize)
		args.secretFile.close()

	if args.action == "local":
		cv = ParallelNestedCV()
		bestItem, innerCVResults = cv.getNestedCVResult(args.classifier, args.datasets, args.optimize)

	if bestItem == None:
		print("Optimization failed")
		sys.exit(1)

	resultDict = OrderedDict()
	resultDict["nestedCVResult"] = bestItem.getDict()
	#some cosmetics on the hyperparameters
	resultDict["innerCVResults"] = [(str(sorted(k.items())).replace("', ", "': "), v.getDict()) for (k,v) in innerCVResults]
	json_output = json.dumps(resultDict, indent = 3)

	if args.outFile:
		args.outFile.write(json_output)
		args.outFile.close()

	if args.json:
		print(json_output)
	else:
		print("Best result after optimization of {} on {}".format(args.classifier, args.datasets))
		if os.isatty(sys.stdout.fileno()):
			print("\033[1m" + bestItem.oneline() + "\033[0m")
		else:
			print(bestItem.oneline())
		print(bestItem)

