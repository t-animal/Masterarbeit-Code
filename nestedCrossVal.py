#!./venv/bin/python
# PYTHON_ARGCOMPLETE_OK

from collections import OrderedDict
from tester import getClassifierClass
from util.argGenerator import generateCrossValidationSets
from util.containers import CrossValidationResultContainer
from util.distribute import ExecutionRequestServer, ExecutionRequestMaster

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

class NestedCV:

	def getCVResults(self, classifierName, dataSets, optimize):
		""" Gets the nested cross validation result for the tuples of optimized hyperparameters
		    and the inner cross validation results for the best-performing optimized hyperparameters


	    :param classifierName: the name of the classifier to use
	    :param dataSets: the name of the data sets to crossvalidate on
	    :param optimize: a dict where each key is a hyperparameter and each value is a list of possible candidates
	    :type classifierName: string
	    :type dataSets: list of strings
	    :type optimize: dict
		"""
		#We could implement a non-parallel version here, but who cares?
		raise NotImplemented()


	@staticmethod
	def performCV(classifierName, dataSets, classifierArgs, includeOuterResults = False):
		""" Performs a nested cross validation and returns a list of the results of the inner cross validations.
		    If includeOuterResults is True, the list contains tuples of the result of the outer validation and
		    the inner cross validation.

		    :param classifierName: the name of the classifier to use
		    :param dataSets: the name of the data sets to crossvalidate on
		    :param classifierArgs: the kw-parameters to pass to the classifier
		    :param includeOuterResults: whether to include the outer results or not
		    :type classifierName: string
		    :type dataSets: list of strings
		    :type classifierArgs: dict
		    :type includeOuterResults: boolean

		    :returns: list of CrossValidateResultContainer or
		              list of tuple (TestResultContainer, CrossValidateResultContiner)
		    """

		testResults = []
		classifier = getClassifierClass(classifierName, "classifiers")(**classifierArgs, )

		for crossTestSet in generateCrossValidationSets(dataSets):
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

	def getCVResults(self, classifierName, dataSets, optimize):
		log.info("Beginning optimization.")
		classifierArgsList = self.getKWTuples(optimize)
		testResultList = self.pool.map(ParallelNestedCV._performCVMapProxy, [(classifierName, dataSets, arg) for arg in classifierArgsList])

		return self.getBestResultsFromParallelComputation(testResultList)

	@staticmethod
	def getBestResultsFromParallelComputation(testResultList):
		"""Similar to the reduce-step of Map-Reduce this method takes a list of results from
		_performCVMapProxy, finds the best argument tuple of the inner CVs for each step
		of the outer CV. It then returns the result of the outer CV based on the pre-computed outerCV
		results for each of the best argument tuples. It also returns a list of tuples of best-
		performing arguments and their results per inner CV.

		:param testResultList: mapped results of _performCVMapProxy
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
	def _performCVMapProxy(args):
		"""Helper method for mapping performCV. It returns a tuple of
		   the passed parameters KW tuples (i.e. the parameters passed to the classifier) and
		   the results of performCV.

			:param args: a tuple consisting of the arguments passed to performCV
			:type args: tuple (string,  list of strings, dict)

			:returns: tuple (dict, tuple (TestresultContainer, CrossValidationResultContainer))
		"""
		return (args[2], NestedCV.performCV(*args, includeOuterResults = True))



class DistributedNestedCV(ParallelNestedCV):

	def __init__(self, secret, workersPort, workers):
		self.distributer = ExecutionRequestMaster(secret, workersPort, workers)

	def getCVResults(self, classifierName, dataSets, optimize):
		kwargsList = self.getKWTuples(optimize)
		testResultList = self.distributer.distribute([(classifierName, dataSets, args) for args in kwargsList])
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
			logging.getLogger("de.t_animal.MA.util.distribute").setLevel(logging.INFO)
		elif args.v > 1:
			logging.basicConfig(level=[logging.WARN, logging.INFO, logging.DEBUG][min(args.v - 1, 2)])
	else:
		logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		                level=[logging.WARN, logging.INFO, logging.DEBUG][min(args.v, 2)])

	if args.action == "worker":
		ExecutionRequestServer.startWorker(args.secretFile.read(), args.port, ParallelNestedCV._performCVMapProxy, args.num)
		args.secretFile.close()
		sys.exit(0)

	if args.model:
		args.optimize["modelPath"] = [modelPaths[args.model]]

	if args.action == "master":
		master = DistributedNestedCV(args.secretFile.read(), args.port, args.workers)
		bestItem, innerCVResults = master.getCVResults(args.classifier, args.datasets, args.optimize)
		args.secretFile.close()

	if args.action == "local":
		cv = ParallelNestedCV()
		bestItem, innerCVResults = cv.getCVResults(args.classifier, args.datasets, args.optimize)

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

