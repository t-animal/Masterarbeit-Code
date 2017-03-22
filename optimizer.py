#!./venv/bin/python

from collections import OrderedDict
from tester import getClassifierClass
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

log = logging.getLogger("de.t_animal.MA.optimizer")

class _signals():
	OK = b'\x00'
	ABORT = b'\x01'

class ExecutionRequestHandler(socketserver.BaseRequestHandler):

	def handle(self):
		poolEpoch = self.server.poolEpoch

		#basic authentication, no need for higher security atm
		salt = "".join([random.choice(string.hexdigits) for x in range(0, 64)])
		authcode = hashlib.pbkdf2_hmac("sha512", self.server.secret.encode("ascii"),
		                               salt.encode("ascii"), 500000)

		self.request.sendall(salt.encode("ascii"))

		if not binascii.hexlify(authcode) == self.request.recv(129):
			log.warning("Requester could not authenticate")
			return

		self.request.sendall(_signals.OK)

		#authenticated, loading arguments
		size = int.from_bytes(self.request.recv(4), byteorder = "big")
		args = pickle.loads(self.request.recv(size))

		log.info("Received computation request: {}({}) on {}".format(args[0], args[2], args[1]))

		result = self.server.pool.apply_async(_getCVResultMapProxy, (args,))

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

		answer = pickle.dumps(result, protocol=3)
		self.request.send(len(answer).to_bytes(4, byteorder="big"))
		self.request.sendall(answer)


class ExecutionRequestServer(socketserver.ThreadingMixIn, socketserver.TCPServer):

	allow_reuse_address = True

	def __init__(self, port, secret, n_cpus):
		super().__init__(("0.0.0.0", port), ExecutionRequestHandler)

		self.secret = secret
		self.n_cpus = n_cpus
		self.pool = multiprocessing.Pool(n_cpus)
		self.poolEpoch = 0
		self.terminationLock = threading.Lock()

	def terminatePool(self, poolEpoch):
		with self.terminationLock:
			if not poolEpoch == self.poolEpoch:
				return

			self.pool.terminate()
			self.pool.join()
			self.pool = multiprocessing.Pool(self.n_cpus)
			self.poolEpoch += 1


def getCVResult(classifierName, crossSet, classifierArgs):

	result = CrossValidationResultContainer("aroused", "nonAroused")

	classifier = getClassifierClass(classifierName)(**classifierArgs)

	for crossTestSet in generateCrossValidationSets(crossSet):
		for crossValidateSet in crossTestSet["crossValidate"]:
			classifier.train(crossValidateSet["train"])
			result.addResult(classifier.test(crossTestSet["test"]))

	log.info(result.oneline())
	return result

def _getCVResultMapProxy(args):
	return (args[2], getCVResult(*args))

def getEfficiencyList(classifierName, crossValidateSet, classifierArgsList):

	pool = multiprocessing.Pool(4)
	resultList = pool.map(_getCVResultMapProxy, [(classifierName, crossValidateSet, arg) for arg in classifierArgsList])

	return OrderedDict(sorted(resultList, key = lambda t: t[1]))


def getKWTuples(p_grid):
	tuples = list(itertools.product(*p_grid.values()))
	kwTuples = []

	for simpleTuple in tuples:
		kwTuples.append({key:simpleTuple[index] for index, key in enumerate(p_grid.keys())})

	return kwTuples


def startWorker(secret, port, n_cpus):
	server = ExecutionRequestServer(port, secret, n_cpus)
	server.serve_forever()


def startMaster(secret, port, workers, classifierName, crossValidateSet, optimize):

	def getWorkerSock(worker):
		try:
			sock = socket.create_connection((worker, port))
		except:
			return None

		salt = sock.recv(64)
		authcode = hashlib.pbkdf2_hmac("sha512", secret.encode("ascii"), salt, 500000)

		sock.sendall(binascii.hexlify(authcode))

		if not sock.recv(1) == _signals.OK:
			log.warning("Could not authenticate on worker %s", worker)
			return None

		#authenticated, adding to list
		return sock

	loopedWorkers = itertools.cycle(workers)
	kwargsList = getKWTuples(optimize)
	dataSets = generateCrossValidationSets(crossValidateSet)

	workerSocks = []
	failedWorkers = []
	assignedTasks = {}

	def dispatchWork(classifierArg):
		worker = next(loopedWorkers)
		workerSock = getWorkerSock(worker)

		while workerSock is None:
			log.warn("Worker %s failed to connect or authenticate", worker)
			failedWorkers.append(worker)
			if set(workers) == set(failedWorkers):
				log.error("All workers have failed!")
				raise KeyboardInterrupt() # maybe some workers have worked before, clean them up

			worker = next(loopedWorkers)
			workerSock = getWorkerSock(worker)

		workerSocks.append((worker, workerSock))
		#assumptions here are: either this fails on all workers or none; all workers have roughly the same speed
		data = pickle.dumps((classifierName, crossValidateSet, classifierArg), protocol=3)
		workerSock.send(len(data).to_bytes(4, byteorder="big"))
		workerSock.sendall(data)
		assignedTasks[workerSock] = classifierArg

	try:
		for classifierArg in kwargsList:
			dispatchWork(classifierArg)

		while len(assignedTasks) > 0:
			results = []
			for index, (worker, workerSock) in enumerate(workerSocks):
				try:
					size = int.from_bytes(workerSock.recv(4), byteorder="big")
					result = pickle.loads(workerSock.recv(size))
				except KeyboardInterrupt:
					raise KeyboardInterrupt()
				except Exception as exception:
					log.warning("Could not get results from worker %s due to exception: %s. Redispatching", worker, exception)
					dispatchWork(assignedTasks[workerSock])
					continue
				finally:
					del assignedTasks[workerSock]

				if result["exception"] is not None:
					log.warning("Worker %s threw an exception: %s. NOT REDISPATCHING!", worker, result["exception"])
				else:
					results.append(result["result"])

	except KeyboardInterrupt:
		log.error("Exiting, sending workers kill signal.")
		log.error("This terminates all computation on all connected workers.")
		log.error("Give the workers a couple of seconds to reinitiate.")
		try:
			for worker, workerSock in workerSocks:
				workerSock.send(_signals.ABORT)
		except BrokenPipeError:
			pass

		return {}

	return OrderedDict([(str(k), v) for k, v in reversed(sorted(results, key=lambda t: t[1]))])



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
								   choices = list(modelPaths.keys()), required = True)
	masterParser.add_argument("--datasets", "-d", help = "The datasets to crossvalidate on (must be available on all cluster workers)",
								   nargs = "+", choices = ChoicesContainer(_dataSets.keys()), required = True)
	masterParser.add_argument("-o", "--outFile", help = "Write json output to this file, too (will be truncated)", type = argparse.FileType("w"))
	masterParser.add_argument("optimize",  help = "The options to optimize as a dict, passed as json (e.g. '{\"C\":[1,2,3], \"foo\":[4.6,5.7]}'", type = json.loads, default = None)


	localParser = commandParsers.add_parser("local", help="like master, but don't connect to any workers, optimize on this machine only",
												parents=[masterParser], conflict_handler='resolve')

	masterParser.add_argument("--workers", "-w",  help = "The clients' hostnames to connect to", type = str, nargs = "+", required = True)

	args = parser.parse_args()


	if args.action == "worker":
		logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level = logging.WARN)
		#if loglevel is 1 print only worker logs, if greater print lower level logs, too
		if args.v == 1:
			log.setLevel(logging.INFO)
		elif args.v > 1:
			logging.basicConfig(level=[logging.WARN, logging.INFO, logging.DEBUG][min(args.v - 1, 2)])

		startWorker(args.secretFile.read(), args.port, args.num)
		args.secretFile.close()
		sys.exit(0)


	logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                level=[logging.WARN, logging.INFO, logging.DEBUG][min(args.v, 2)])

	if args.action == "master":
		args.optimize["modelPath"] = [modelPaths[args.model]]
		result = startMaster(args.secretFile.read(), args.port, args.workers, args.classifier, args.datasets, args.optimize)

	if args.action == "local":
		args.optimize["modelPath"] = [modelPaths[args.model]]
		result = getEfficiencyList(args.classifier, args.datasets, getKWTuples(args.optimize))

	if result == {}:
		print("Optimization failed")
		sys.exit(1)

	json_output = json.dumps({k: v.getDict() for k, v in result.items()})

	bestItem = result.popitem(last = False)

	if args.outFile:
		args.outFile.write(json_output)
		args.outFile.close()

	if args.json:
		print(json_output)
	else:
		print("Best result after optimization of {} on {}")
		if os.isatty(sys.stdout.fileno()):
			print("\033[1m" + bestItem[1].oneline() + "\033[0m")
		else:
			print(bestItem[1].oneline())
		print(bestItem[1])
		print("For these hyperparameters: {}".format(bestItem[0]))

