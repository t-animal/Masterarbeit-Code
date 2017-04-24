"""
NN trainer for word embeddings (wiki)
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2016 Feb 9th"

from . import processors
from . import layers

import numpy
import signal

from threading import Thread

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


# NO_DATA = 3000000


NO_DATA = 6


"""
WeightLearningNetwork trains a small neural network in which we assign a weight to each vector in the sentence, normalize and cutt off.
We learn the weights directly through linear interpolation.
"""

def cutoff(x):
    return 0.0
    #return 0.92258 * numpy.exp(-0.27144 * x) + 0.09834

class WeightLearningNetwork():
    def __init__(self, wordvectors, maxNoWords = 30, batch_size = 100):
        """
           :param wordvectors: the word vector model to use
           :param maxNoWords: the maximum number of words a document may have
           :type wordvectors: gensim.model.KeyedVectors
           :type maxNoWords: int
        """

        self.wordvectors = wordvectors
        self.embeddings_dim = wordvectors[wordvectors.index2word[0]].size
        self.maxNoWords = maxNoWords
        self.batch_size = batch_size

        input_shape = (self.embeddings_dim, self.maxNoWords)
        output_shape = (self.embeddings_dim, 1)

        self.numpy_rng = numpy.random.RandomState(89677)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        self.x1 = T.tensor3('x1')
        self.x2 = T.tensor3('x2')
        self.indices1 = T.matrix('i1')
        self.indices2 = T.matrix('i2')
        self.l1 = T.vector('l1')
        self.l2 = T.vector('l2')
        self.y = T.vector('y') #0 or 1
        self.z = T.vector('z') #-1 or 1

        self.model1 = layers.interpolatingDotMeanLayer(self.numpy_rng, self.theano_rng, self.x1, input_shape,
                                                             self.indices1, self.l1, max_length=35, n_out=output_shape[1], batch_size=batch_size, W=None)
        self.model2 = layers.interpolatingDotMeanLayer(self.numpy_rng, self.theano_rng, self.x2, input_shape,
                                                             self.indices2, self.l2, max_length=35, n_out=output_shape[1], batch_size=batch_size, W=self.model1.W)

        self.params = []
        self.params.extend(self.model1.params)

    def run(self, pairs, noPairs, docFreq, epochs = 3, learning_rate = 1.0, regularization = 0.001, momentum = 0.0, cost_factor = 160.0):
        """
            :param pairs: list of documents which are related; each item must be two related documents seperated by semicolon
            :param noPairs: list of documents which are unrelated; each item must be two unrelated documents seperated by semicolon
            :param docFreq: document frequency of words in the given model
            :param epochs:  ???
            :param learning_rate: ???
            :param regularization: ???
            :param momentum: ???
            :param cost_factor: ???

            :type pairs: list of strings
            :type noPairs: list of strings
            :type docFreq: list of ints (use functions in documentFrequency for further reference)
            :type epochs: int
            :type learning_rate: float
            :type regularization: float
            :type momentum: float
            :type cost_factor: float

            :returns: numpy.array
        """

        #FIXME: Is this correct? it's a guess from the processors.py's code
        NO_DATA = min(len(pairs), len(noPairs))*2

        processor = processors.LengthTweetPairProcessor(pairs, noPairs, docFreq, self.wordvectors,
                                                            self.maxNoWords, self.embeddings_dim, self.batch_size, cutoff)
        train_x1 = theano.shared(value=processor.x1, name='train_x1', borrow=False)
        train_x2 = theano.shared(value=processor.x2, name='train_x2', borrow=False)
        train_i1 = theano.shared(value=processor.indices1, name='train_i1', borrow=False)
        train_i2 = theano.shared(value=processor.indices2, name='train_i2', borrow=False)
        train_l1 = theano.shared(value=processor.l1, name='train_l1', borrow=False)
        train_l2 = theano.shared(value=processor.l2, name='train_l2', borrow=False)
        train_y = theano.shared(value=processor.y, name='train_y', borrow=False)
        train_z = theano.shared(value=processor.z, name='train_z', borrow=False)

        print('Initializing train function...')
        train = self.train_function_momentum(train_x1, train_x2, train_i1, train_i2, train_l1, train_l2, train_y, train_z)

        print('Cost factor: ' + str(cost_factor))

        t = Thread(target=processor.process)
        t.daemon = True
        t.start()

        def signal_handler(signal, frame):
            import os
            os._exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        batches = int(NO_DATA / self.batch_size) #FIXME this cuts off (as does the original implementation), is this correct?
        best_cost = float('inf')
        best_weights = None
        previous_best_cost = float('inf')
        second_time = False

        for e in range(epochs):
            processor.new_epoch()

            processor.lock.acquire()
            while not processor.ready:
                processor.lock.wait()
            processor.lock.release()

            train_x1.set_value(processor.x1, borrow=False)
            train_x2.set_value(processor.x2, borrow=False)
            train_i1.set_value(processor.indices1, borrow=False)
            train_i2.set_value(processor.indices2, borrow=False)
            train_l1.set_value(processor.l1, borrow=False)
            train_l2.set_value(processor.l2, borrow=False)
            train_y.set_value(processor.y, borrow=False)
            train_z.set_value(processor.z, borrow=False)

            processor.lock.acquire()
            processor.cont = True
            processor.ready = False
            processor.lock.notifyAll()
            processor.lock.release()

            for b in range(batches):
                cost = train(lr=learning_rate, reg=regularization, mom=momentum)

                processor.lock.acquire()
                while not processor.ready:
                    processor.lock.wait()
                processor.lock.release()

                print('Training, batch %d (from %d), cost %.5f' % (b, batches, cost))
                we = self.model1.W.get_value()
                print(repr(we))

                train_x1.set_value(processor.x1, borrow=False)
                train_x2.set_value(processor.x2, borrow=False)
                train_i1.set_value(processor.indices1, borrow=False)
                train_i2.set_value(processor.indices2, borrow=False)
                train_l1.set_value(processor.l1, borrow=False)
                train_l2.set_value(processor.l2, borrow=False)
                train_y.set_value(processor.y, borrow=False)
                train_z.set_value(processor.z, borrow=False)

                processor.lock.acquire()
                if b < batches-2:
                    processor.cont = True
                    processor.ready = False
                if b == batches-1 and e == epochs-1:
                    processor.stop = True
                    processor.cont = True
                processor.lock.notifyAll()
                processor.lock.release()

            #print('Training, factor %d, lr %.5f, epoch %d, cost %.5f' % (int(cost_factor), learning_rate, e, numpy.mean(c)))
            #we = self.model1.W.get_value()
            #print(repr(we))

            # UNCOMMENT WHEN THIS PIECE OF CODE IS CALLED EXTERNALLY
            if numpy.mean(cost) < best_cost - 0.0005:
                previous_best_cost = best_cost
                best_cost = numpy.mean(cost)
                best_weights = we
            elif second_time:
                processor.lock.acquire()
                processor.stop = True
                processor.cont = True
                processor.lock.notifyAll()
                processor.lock.release()
                break
            else:
                best_cost = previous_best_cost
                learning_rate = 0.001
                second_time = True

        t.join()

        print("Best weights:")
        print(repr(best_weights))
        return best_weights

    def save_me(self, filename=None):
        self.model1.save_me(filename)

    def load_me(self, filename=None):
        self.model1.load_me(filename)
        self.model2.W = self.model1.W

    def train_function_momentum(self, x1, x2, i1, i2, l1, l2, y, z):
        """Train model with momentum"""

        learning_rate = T.scalar('lr')  # learning rate to use
        regularization = T.scalar('reg')  # regularization to use
        momentum = T.scalar('mom')  # momentum to use

        cost, updates = self.get_cost_updates_momentum(learning_rate, regularization, momentum)

        train_fn = theano.function(
            inputs=[
                theano.Param(learning_rate, default=0.1),
                theano.Param(regularization, default=0.0),
                theano.Param(momentum, default=0.9)
            ],
            outputs=cost,
            updates=updates,
            givens={
                self.x1: x1,
                self.x2: x2,
                self.indices1: i1,
                self.indices2: i2,
                self.l1: l1,
                self.l2: l2,
                self.y: y,
                self.z: z
            },
            name='train_momentum',
            on_unused_input='warn'
        )

        return train_fn

    def get_cost_updates_momentum(self, learning_rate, regularization, momentum):
        """Calculate updates of params based on momentum gradient descent"""

        cost0 = self.calculate_cost()
        cost = cost0 + regularization * self.calculate_regularization()
        gparams = T.grad(cost, self.params)

        updates = []
        for p, g in zip(self.params, gparams):
            mparam_i = theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = momentum * mparam_i - learning_rate * g
            updates.append((mparam_i, v))
            updates.append((p, p + v))

        return (cost0, updates)

    def calculate_cost(self):
        output1 = self.model1.output
        output2 = self.model2.output

        ## Euclidean loss
        loss = ((output1 - output2) ** 2).sum(axis=1) * (-self.z)  #--> EUCLIDEAN LOSS
        return T.mean(loss)

        ### Median loss with cross-entropy
        # distances = ((output1 - output2) ** 2).sum(axis=1)
        # sorted_distances = T.sort(distances)
        # median = (sorted_distances[self.batch_size/2] + sorted_distances[self.batch_size/2 - 1]) / 2.0
        #
        # p = distances[0:self.batch_size:2]  #pairs
        # q = distances[1:self.batch_size:2]  #non-pairs
        #
        # loss = (T.log(1.0 + T.exp(-cost_factor*(q - median)))).mean() + \
        #     (T.log(1.0 + T.exp(-cost_factor*(median - p)))).mean()          #cross-entropy
        # return loss

    def calculate_regularization(self):
        return (self.model1.W ** 2).sum()


if __name__ == '__main__':
    n = WeightLearningNetwork()
    n.run()