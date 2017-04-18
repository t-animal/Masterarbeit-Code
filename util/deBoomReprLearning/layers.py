import numpy

import theano
import theano.tensor as T


class interpolatingDotMeanLayer():
    """Layer calculating dot product of input with interpolated weights and taking the mean"""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, indices, length, max_length=30, n_out=1, batch_size=100, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.indices = indices #2D tensor
        self.length = length #1D tensor
        self.max_length = float(max_length)
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        init_W = [ 0.54457003,  0.72741562,  1.39331913,  1.12367916,  0.79878163,
        0.27706152,  0.3593896 ,  0.39622781,  0.27895978,  0.23260947,
        0.26763204,  0.27084899,  0.07067534,  0.13463201,  0.07948229,
        0.02779013,  0.12053657,  0.14807181,  0.24277158,  0.36964679,
        0.1601541 ,  0.37342793,  0.47257897,  0.39729786,  0.56589139,
        0.30535939,  0.10021771,  0.07151619,  0.12510002,  0.3112531 ,
        0.43562451,  0.05050614,  0.07199406,  0.50659907,  0.42588547]

        if W is None:
            W_values = numpy.asarray(
                    self.numpy_rng.uniform(
                        low=0.5,
                        high=0.5,
                        size=(self.n_in)
                    ),
                    #init_W,
                    # numpy.linspace(1.0, 0.0, self.n_in),
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.indices_high = T.ceil(self.indices).astype('int8')
        self.indices_low = T.floor(self.indices).astype('int8')
        self.factors_high = self.W[self.indices_high]
        self.factors_low = self.W[self.indices_low]
        self.factors = (self.factors_high - self.factors_low) * (self.indices - self.indices_low) / \
                       (self.indices_high - self.indices_low + 1E-5) + self.factors_low
        self.output = T.sum(self.x * T.transpose(self.factors).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')

        self.params = [self.W]

    def save_me(self, filename=None):
        f = open(filename, 'wb')
        numpy.savez_compressed(f, self.W.get_value(borrow=True))
        f.close()

    def load_me(self, filename=None):
        f = open(filename, 'rb')
        dictionary = numpy.load(f)
        self.W.set_value(dictionary['arr_0'])
        f.close()

    def get_hidden_values(self, input, batch_size):
        self.indices_high = T.ceil(self.indices).astype('int8')
        self.indices_low = T.floor(self.indices).astype('int8')
        self.factors_high = self.W[self.indices_high]
        self.factors_low = self.W[self.indices_low]
        self.factors = (self.factors_high - self.factors_low) * (self.indices - self.indices_low) / \
                       (self.indices_high - self.indices_low + 1E-5) + self.factors_low
        self.output = T.sum(self.x * T.transpose(self.factors).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')
