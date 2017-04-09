import binascii
import hashlib
import itertools
import logging
import multiprocessing
import pickle
import random
import socket
import socketserver
import string
import threading

from util import set_keepalive_linux

log = logging.getLogger("de.t_animal.MA.util.distribute")

class _signals():
	OK = b'\x00'
	ABORT = b'\x01'

#This should be around 500000
N_ROUNDS=500

class ExecutionRequestHandler(socketserver.BaseRequestHandler):
	def handle(self):
		"""Handles an execution request passed on by a ExecutionRequestServer (aka worker).

		Communication with the master is as follows:
			1) Worker sends a random, 64byte string
			2) Server sends the pbkdf_hmac of the shared secret using the random string as salt
			3) Worker sends the _signals.OK or closes connection
			4) Server sends a 4-byte integer I in Big Endian indicating the length of the following information
			5) Server sends the pickled computation request of size I (in bytes)
			6) Worker passes the computation request to ParallelNestedCV._performCVMapProxy
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

		result = self.server.pool.apply_async(self.server.workerFunction, (args,))

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

	def __init__(self, port, secret, workerFunction, n_cpus):
		""" Usually not needed, use startWorker to start a worker and let it run until killed.

		:param port: the port to listen on
		:param secret: the secret to secure the connection to masters with
		:param workerFunction: the function to execute
		:param n_cpus: the maximum amount of processes to start
		:type port: integer
		:type secret: string
		:type n_cpus: integer > 0
		"""
		super().__init__(("0.0.0.0", port), ExecutionRequestHandler)

		self.secret = secret
		self.n_cpus = n_cpus
		self.workerFunction = workerFunction
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
	def startWorker(secret, port, workerFunction, n_cpus):
		""" Starts a worker and serves forever. Does not return unless an Exception is raised!

		:param port: the port to listen on
		:param secret: the secret to secure the connection to masters with
		:param workerFunction: the function to execute
		:param n_cpus: the maximum amount of processes to start
		:type port: integer
		:type secret: string
		:type n_cpus: integer > 0
		"""
		server = ExecutionRequestServer(port, secret, workerFunction, n_cpus)
		server.serve_forever()


class ExecutionRequestMaster:
	""" A client connecting to many worker servers, passing them arguments to compute on. The function
	    which is executed is determined by the workers (i.e. by the user starting either a NestedCV or
	    CV worker)! Only the arguments are sent via the network (in contrast to usual RPC where the
	    function is pickled and sent, too).
	"""

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

	def dispatchWork(self, workerFunctionArgs):
		"""Dispatches a work assignment to the next worker.

	    :param workerFunctionArgs: the arguments to pass to the workerfunction
	    :type workerFunctionArgs: tuple
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
		data = pickle.dumps(workerFunctionArgs, protocol=3)
		workerSock.send(len(data).to_bytes(4, byteorder="big"))
		workerSock.sendall(data)

		self.assignedTasks[workerSock] = workerFunctionArgs


	def distribute(self, workerFunctionArgsList):
		try:
			log.info("Beginn dispatching work assignments")
			for workerFunctionArgs in workerFunctionArgsList:
				self.dispatchWork(workerFunctionArgs)
			log.info("Dispatched all work assignments")

			testResultList = []
			while len(self.assignedTasks) > 0:
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
						self.dispatchWork(self.assignedTasks[workerSock])
						del self.workerSocks[index]
						del self.assignedTasks[workerSock]
						continue

					del self.workerSocks[index]
					del self.assignedTasks[workerSock]

					if result["exception"] is not None:
						log.warning("Worker %s threw an exception: %s. NOT REDISPATCHING!", worker, result["exception"])
					else:
						testResultList.append(result["result"])
						log.info("Worker %s has finished a computation (%d/%d done)", worker, len(testResultList), len(workerFunctionArgsList))

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

		return testResultList
