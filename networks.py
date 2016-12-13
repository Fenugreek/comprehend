"""
TensorFlow implementations some NN architectures.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
from numpy.random import randint, random_sample, randn
import cPickle
from collections import OrderedDict
from scipy.stats import multivariate_normal

from tamarind.functions import sigmoid, unit_scale, logit
from tamarind import arrays
import functions, train

float_dt = tf.float32

def xavier_init(fan_in, fan_out, name='W', constant=1,
                shape=None, trainable=True, dtype=float_dt):
    """
    Xavier initialization of network weights.

    See https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    """
    bound = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.Variable(tf.random_uniform((fan_in, fan_out) if shape is None else shape,
                                         minval=-bound, maxval=bound, 
                                         dtype=dtype, name=name),
                       trainable=trainable)


class Coder(object):
    """Virtual NN with skeleton implementation of common functions."""

    param_names = []
    attr_names = ['n_visible', 'n_hidden']

    def __init__(self, n_visible=784, n_hidden=500, verbose=False,
                 random_seed=123, params=None, fromfile=None, dtype=float_dt,
                 coding=tf.sigmoid, decoding=None, trainable=True, **kwargs):
        """
        Initialize the object by specifying the number of visible units (the
        dimension d of the input), the number of hidden units (the dimension
        d' of the latent or hidden space).

        :type random_seed: int
        :param random_seed: seed random number generator with this to get
                            reproducible results.

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type verbose: bool
        :param verbose:  whether to print certain log messages.

        :type fromfile: str or filehandle
        :param fromfile: initialize params from this saved file

        :type coding: function
        :param coding: activation function for converting linear combination of
                       visible units' values to hidden value.

        :type decoding: function
        :param decoding: activation function for converting linear combination of
                         hidden units' values back to visible value.
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.coding = coding
        self.decoding = coding if decoding is None else decoding
        self.verbose = verbose
        self.dtype = dtype
        
        if random_seed is not None: tf.set_random_seed(random_seed)

        if params is not None:
            self.params = params
        else: 
            self.params = OrderedDict()
            if fromfile is not None:
                is_handle = type(fromfile) == file
                save_file = fromfile if is_handle else open(fromfile)

                for name in type(self).param_names:
                    self.params[name] = tf.Variable(cPickle.load(save_file),
                                                    name=name, trainable=trainable)
                try:
                    for attr in type(self).attr_names:
                        setattr(self, attr, cPickle.load(save_file))
                except EOFError: pass
    
                if not is_handle: save_file.close()
            else:
                self.init_params(**kwargs)
            
        self.init_train_args(**kwargs)
        if not hasattr(self, 'output_dims'):
            self.output_dims = self.input_dims
        

    def init_params(self, **kwargs):
        pass

    def init_train_args(self, **kwargs):
        # To be used for training by tf.Optimizer objects.
        self.train_args = [tf.placeholder(self.dtype,
                                          shape=[None, self.n_visible])]
        self.input_dims = 2

    
    def input_shape(self):
        return [getattr(self, 'batch_size', -1)] + [self.n_visible]


    def output_shape(self, **kwargs):
        return [getattr(self, 'batch_size', -1)] + [self.n_hidden]


    def save_params(self, tofile):
        """
        Save params to disk, compatible with fromfile option of constructor.
        If argument is a string, a new file is with that name is written to.
        If argument is a file handle, data is written to that.
        """

        is_handle = type(tofile) == file
        save_file = tofile if is_handle else open(tofile, 'wb')
        
        for variable in self.params.values():
            cPickle.dump(variable.eval(), save_file, -1)

        for attr in type(self).attr_names:
            cPickle.dump(getattr(self, attr), save_file, -1)

        if not is_handle: save_file.close()    


    def _get_set_batch_size(self, batch_size=None):

        if batch_size is None: 
            if hasattr(self, 'batch_size'): return self.batch_size
        elif hasattr(self, 'set_batch_size'):
            self.set_batch_size(batch_size)
            
        return batch_size


    def dump_output(self, data, filename, kind='hidden',
                    batch_size=None, dtype=None, **kwargs):
        """
        Compute hidden/recode values for given input data, according to current
        weights of the network, and then write to disk.

        Useful to perform training on subsequent layer ('hidden'), or
        comparison of recoded input with original ('recode').

        Output is computed one batch at a time, if batch_size is given or
        self has attribute batch_size.

        kind:
        'hidden' or 'recode'.
        Values of hidden units, or reconstructed visible units
        
        batch_size:
        Compute values for these many rows at a time, to save memory,
        saving results one batch at a time.
        If <self> has <batch_size> as an attribute, that is used by default.
        """
        
        if kind == 'hidden': kind = 'get_hidden_values'
        method = getattr(self, kind, None)
        if method is None:
            raise ValueError('Do not understand kind option: ' + kind)

        save_file = open(filename, 'wb')
        batch_size = self._get_set_batch_size(batch_size) or len(data)
                
        for i in range(0, len(data) - batch_size + 1, batch_size):
            values = method(data[i:i+batch_size], **kwargs).eval()
            if dtype is not None: values = values.astype(dtype, copy=False)
            cPickle.dump(values, save_file, -1)
                       
        save_file.close()    
        

    def cost_args(self, dataset):
        """Return args to self.cost() for given dataset."""
        return [dataset]


    def train_feed(self, data):
        """Return feed_dict based on data to be used for training."""
        return {self.train_args[0]: data}


    def get_hidden_values(self, inputs, **kwargs):
        """Computes the values of the hidden layer."""
        return None #implement in subclass
    
        
    def get_reconstructed_input(self, hidden, **kwargs):
        """
        Computes the reconstructed input given the values of the
        hidden layer.
        """
        return None #implement in subclass


    def recode(self, inputs, **kwargs):
        """Encode and then decode input, using current params."""
        y = self.get_hidden_values(inputs, **kwargs)
        return self.get_reconstructed_input(y, **kwargs)


    def cost(self, inputs, function=functions.cross_entropy, **kwargs):
        """
        Cost for given input batch of samples, under current params.
        Mean cross entropy between input and encode-decoded input.
        """
        loss = function(inputs, self.recode(inputs, **kwargs))
        return tf.reduce_mean(
                        tf.reduce_sum(loss,
                                      reduction_indices=range(1, self.input_dims)))


    def label_cost(self, inputs, labels):
        """
        For classification problems, mean cross entropy between class probabilities.
        i.e. Cost for given input batch of samples, under current params.
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.get_hidden_values(inputs), labels)
        return tf.reduce_mean(loss)


    def target_cost(self, inputs, targets, function=tf.squared_difference):
        """
        For mapping problems, r.m.s. difference between hidden values and targets.
        i.e. Cost for given input batch of samples, under current params.
        """
        return tf.reduce_mean(function(self.get_hidden_values(inputs), targets))


    def rms_loss(self, inputs, **kwargs):
        """
        Root-mean-squared difference between <inputs> and encoded-decoded output.
        """
        loss = tf.squared_difference(inputs, self.recode(inputs, **kwargs))
        return tf.reduce_mean(
                        tf.reduce_mean(loss,
                                       reduction_indices=range(1, self.input_dims)) ** .5)


class Auto(Coder):
    """
    Auto-Encoder. Adapted from deeplearning.net.

    :type W: tf.Variable
    :param W: TensorFlow variable pointing to a set of weights that should be
              shared between this object and another architecture;
              if object is standalone set to None.

    :type bhid: tf.Variable
    :param bhid: TensorFlow variable pointing to a set of biases values (for
                 hidden units) that should be shared betweeen this object and
                 another architecture; if object is standalone set to None.

    :type bvis: tf.Variable
    :param bvis: TensorFlow variable pointing to a set of biases values (for
                 visible units) that should be shared betweeen this object
                 and another architecture; if obj is standalone set to None.
    """

    param_names = ['W', 'bhid', 'bvis']


    def init_params(self, trainable=True, **kwargs):

        self.params['W'] = xavier_init(self.n_visible, self.n_hidden,
                                       name='W', trainable=trainable, dtype=self.dtype)
        self.params['bhid'] = tf.Variable(tf.zeros([self.n_hidden], dtype=self.dtype),
                                          name='bhid', trainable=trainable)
        self.params['bvis'] = tf.Variable(tf.zeros([self.n_visible], dtype=self.dtype),
                                          name='bvis', trainable=trainable)

        
    def get_hidden_values(self, inputs, **kwargs):
        """Computes the values of the hidden layer."""
        return self.coding(tf.matmul(inputs, self.params['W']) +
                           self.params['bhid'])


    def get_reconstructed_input(self, hidden, **kwargs):
        """
        Computes the reconstructed input given the values of the
        hidden layer.
        """
        return self.decoding(tf.matmul(hidden, self.params['W'], transpose_b=True) +
                             self.params['bvis'])


    def features(self, *args):
        """Return weights in suitable manner for plotting."""
        return self.params['W'].eval().T


    def invert(self):
        """Invert weights, biases so visible <=> hidden."""

        self.params['W'] = tf.transpose(self.params['W'])
        
        bhid = self.params['bhid']
        self.params['bhid'] = self.params['bvis']
        self.params['bvis'] = bhid
        
        n_hidden = self.n_hidden
        self.n_hidden = self.n_visible
        self.n_visible = n_hidden

        
class Denoising(Auto):
    """
    Denoising Auto-Encoder.
    
    Same as Auto-Encoder except cost is computed by encode-decoding
    a corrupted copy of the input.
    """
    

    def __init__(self, corruption=0.3, **kwargs):
        """
        corruption:
        Fraction of input elements chosen at random,
        and set to the mean value of input,
        before encode-decoding for the purpose of cost calculation.
        """

        Auto.__init__(self, **kwargs)

        self.corruption = corruption

        # Train args has an additional element:
        #   the corrupted version of the input.
        self.train_args.append(tf.placeholder(self.dtype,
                                              shape=[None, self.n_visible]))
        

    def cost(self, inputs, corrupts, **kwargs):
        """
        Cost for given input batch of samples, under current params.
        Mean cross entropy between input and encode-decoded corrupted
        version of the input.
        """
        loss = functions.cross_entropy(inputs, self.recode(corrupts, **kwargs))
        return tf.reduce_mean(
                        tf.reduce_sum(loss,
                                      reduction_indices=range(1, self.input_dims)))


    def cost_args(self, data):
        """Return args to self.cost() for given dataset."""
        return [data, train.corrupt(data, self.corruption)]


    def train_feed(self, data):
        """Return feed_dict based on data to be used for training."""
        return {self.train_args[0]: data,
                self.train_args[1]: train.corrupt(data, self.corruption)}



class Conv(Coder):
    """
    Convolutional Auto-Encoder.
    """
    
    param_names = ['W', 'bhid', 'bvis']
    attr_names = Coder.attr_names + ['shapes', 'strides', 'padding']

    @staticmethod
    def prep_mnist(data):
        """
        Reshape MNIST data from batch x pixels to
        batch x columns x rows x channels.
        """
        return data.reshape((-1, 28, 28, 1)).swapaxes(1, 2)


    def __init__(self, input_shape=[28, 28, 1], kernel_shape=[5, 5, 1],
                 strides=[1, 1], batch_size=100, coding=tf.nn.elu,
                 padding='SAME', **kwargs):
        """
        input_shape, kernel_shape:
        (rows, columns) of input data and convolution kernel respectively.

        n_hidden:
        Number of feature maps to use.

        batch_size:
        batch size to expect during gradient descent training
        (computation graph needs this ahead of time).
        """
        
        self.batch_size = batch_size
        if not kwargs.get('fromfile'):
            self.strides = strides
            self.shapes = [input_shape, kernel_shape]
            kwargs['n_visible'] = np.prod(input_shape)
            self.padding = padding

        Coder.__init__(self, coding=coding, **kwargs)

        # add default no. of channels, if unspecified
        if len(self.shapes[0]) == 2: self.shapes[0].append(1)
        if len(self.shapes[1]) == 2: self.shapes[1].append(self.shapes[0][2])
        # also strides...
        if len(self.strides) == 2: self.strides = [1] + self.strides + [1]


    def input_shape(self):
        return [self.batch_size] + self.shapes[0]


    def output_shape(self, **kwargs):
        if self.padding == 'SAME':
            dims = [self.shapes[0][i] / self.strides[i+1] for i in range(2)]
        elif self.padding == 'VALID':
            k_shapes = self.shapes[1][:2]
            if hasattr(self, 'rate'):
                k_shapes = [(s - 1) * (self.rate - 1) + s for s in k_shapes]
            dims = [1 + (self.shapes[0][i] - k_shapes[i]) / self.strides[i+1]
                    for i in range(2)]
        return [self.batch_size] + dims + [self.n_hidden]


    def init_train_args(self, **kwargs):
        # To be used for training by tf.Optimizer objects.
        self.train_args = [tf.placeholder(self.dtype,
                                          shape=[None] + self.shapes[0])]
        self.input_dims = 4


    def init_params(self, trainable=True):

        i_shape, k_shape = self.shapes

        # Compute effective number of neurons per filter. Ignores padding.
        conv_out = i_shape[0] * i_shape[1]
        if hasattr(self, 'pool_side'): conv_out /= self.pool_side**2
        elif hasattr(self, 'pool_width'): conv_out /= self.pool_width
        
        self.params['W'] = xavier_init(self.n_visible, self.n_hidden * conv_out,
                                       shape=k_shape + [self.n_hidden],
                                       name='W', trainable=trainable, dtype=self.dtype)
        self.params['bhid'] = tf.Variable(tf.zeros([self.n_hidden], dtype=self.dtype),
                                          name='bhid', trainable=trainable)
        self.params['bvis'] = tf.Variable(tf.zeros(i_shape, dtype=self.dtype),
                                          name='bvis', trainable=trainable)


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        

    def get_hidden_values(self, inputs, **kwargs):

        h_conv = tf.nn.conv2d(inputs, self.params['W'],
                              strides=self.strides, padding=self.padding)
        return self.coding(h_conv + self.params['bhid'])


    def get_reconstructed_input(self, hidden, scale=None, **kwargs):
        
        shape = [self.batch_size] + self.shapes[0]
        outputs = tf.nn.conv2d_transpose(hidden, self.params['W'], shape,
                                         self.strides, padding=self.padding)
        if scale: outputs *= scale
        
        return self.decoding(outputs + self.params['bvis'])


    def features(self, *args):
        """Return n_hidden number of kernel weights."""
        
        return np.rollaxis(self.params['W'].eval(), 3)


class ConvMaxSquare(Conv):
    """
    Convolutional Auto-Encoder with max pooling, on non-overlapping
    square blocks.
    """
    attr_names = Conv.attr_names + ['pool_side']

    def __init__(self, pool_side=2, **kwargs):
        """
        pool_side:
        Do max pooling on pool_side x pool_side non-overlapping
        patches of input.
        """

        Conv.__init__(self, **kwargs)

        if not kwargs.get('fromfile'):
            self.pool_side = pool_side
            self.shapes.append([])

        # Pool shape
        input_size = self.shapes[0] if self.padding == 'SAME' else \
                     [self.shapes[0][i] - self.shapes[1][i] + 1 for i in range(2)]
        self.shapes[2] = [self.batch_size] + \
                         [input_size[i] / self.strides[i+1] /
                          self.pool_side for i in range(2)] + \
                          [self.pool_side**2, self.n_hidden]
        self.zeros = tf.zeros(self.shapes[2], dtype=self.dtype)
        self.state = {}


    def set_batch_size(self, batch_size):

        if self.batch_size == batch_size: return
        self.batch_size = batch_size
        self.shapes[2][0] = batch_size
        self.zeros = tf.zeros(self.shapes[2], dtype=self.dtype)
        self.state = {}
        

    def output_shape(self, reduced=True):
        if reduced: return self.shapes[2][:3] + [self.n_hidden]
        else:       return Conv.output_shape(self)


    def get_hidden_values(self, input, reduced=False, store=None, **kwargs):
        """
        Return hidden values after max pooling.
        
        reduced:
        Values not maximal are set to 0 if reduced is False.
        If reduced is True, values not maximal are omitted.

        store:
        If True, store locations of maximal values.
        Necessary to reconstruct input if reduced is set to True.
        Defaults to value of reduced.
        """

        if store is None: store = reduced
        
        hidden = Conv.get_hidden_values(self, input)
        pool = tf.reshape(tf.space_to_depth(hidden, self.pool_side),
                          self.shapes[2])

        if reduced and not store: return tf.reduce_max(pool, 3)
                    
        # Replace all non-max values with 0.0.
        overlay = tf.one_hot(tf.argmax(pool, 3), self.pool_side**2,
                             axis=3, on_value=True, off_value=False)
        if store: self.state['overlay'] = overlay

        return tf.reduce_max(pool, 3) if reduced else \
               self._pool_overlay(pool, overlay)


    def _pool_overlay(self, pool, overlay):
        
        pool = tf.select(overlay, pool, self.zeros)
        pool_shape = self.shapes[2]        
        return tf.depth_to_space(tf.reshape(pool, pool_shape[:3] + \
                                    [pool_shape[3] * pool_shape[4]]),
                                 self.pool_side)


    def get_reconstructed_input(self, hidden, reduced=False, **kwargs):

        if not reduced: return Conv.get_reconstructed_input(self, hidden)

        hidden = tf.tile(tf.expand_dims(hidden, 3),
                         [1, 1, 1, self.pool_side**2, 1])
        return Conv.get_reconstructed_input(self,
                                            self._pool_overlay(hidden,
                                                               self.state['overlay']))
        
        
class ConvMax1D(ConvMaxSquare):
    """
    Convolutional Auto-Encoder with max pooling, using a kernel shape
    that covers height of input completely.
    
    So results of convolution is 1D, and max pooling is done on this 1D.
    """
    attr_names = Conv.attr_names + ['pool_width']

    def __init__(self, input_shape=[28, 28, 1],
                 kernel_width=5, pool_width=2, **kwargs):
        """
        input_shape, axis:
        (rows, columns) of input data and axis along which convolution kernel
        slides.

        kernel_width, pool_width:
        The 1D size of the kernel along the dimension to slide, and
        the 1D size of the max pooling window.
        """

        Conv.__init__(self, input_shape=input_shape,
                      kernel_shape=[kernel_width] + input_shape[1:], **kwargs)

        if not kwargs.get('fromfile'):
            self.pool_width = pool_width
            self.shapes.append([])
            
        # Pool shape
        input_size = self.shapes[0][0] if self.padding == 'SAME' else \
                     self.shapes[0][0] - self.shapes[1][0] + 1
        self.shapes[2] = [self.batch_size,
                          input_size / self.strides[1] / self.pool_width,
                          self.pool_width, 1, self.n_hidden]
        self.zeros = tf.zeros(self.shapes[2], dtype=tf.float32)
        self.state = {}


    def output_shape(self, reduced=True):
        if reduced: return self.shapes[2][:2] + [1, self.n_hidden]
        else:       return Conv.output_shape(self)


    def get_hidden_values(self, inputs, reduced=False, store=False, **kwargs):

        hidden = Conv.get_hidden_values(self, inputs)

        pool = tf.reshape(hidden, self.shapes[2])
        if reduced and not store: return tf.reduce_max(pool, 2)

        # Replace all non-max values with 0.0.
        overlay = tf.one_hot(tf.argmax(pool, 2), self.pool_width,
                             axis=2, on_value=True, off_value=False)
        if store: self.state['overlay'] = overlay

        return tf.reduce_max(pool, 2) if reduced else \
               self._pool_overlay(pool, overlay)


    def _pool_overlay(self, pool, overlay):

        pool = tf.select(overlay, pool, self.zeros)
        pool_shape = self.shapes[2]
        return tf.reshape(pool, [self.batch_size, pool_shape[1] * pool_shape[2],
                                 1, self.n_hidden])


    def _make_overlay(self, location):
        if np.isscalar(location): location = [location]
        overlay = np.zeros(self.shapes[2], np.bool)
        for loc in location:
            overlay[:, :, loc, ...] = True
        return overlay


    def _random_overlay(self, static_hidden=False):
        """Construct random max pool locations."""

        s = self.shapes[2]

        if static_hidden:
            args = np.random.randint(s[2], size=np.prod(s) / s[2] / s[4])
            overlay = np.zeros(np.prod(s) / s[4], np.bool)
            overlay[args + np.arange(len(args)) * s[2]] = True
            overlay = overlay.reshape([s[0], s[1], s[3], s[2]])
            overlay = np.rollaxis(overlay, -1, 2)
            return tamarind.arrays.extend(overlay, s[4])
        else:
            args = np.random.randint(s[2], size=np.prod(s) / s[2])
            overlay = np.zeros(np.prod(s), np.bool)
            overlay[args + np.arange(len(args)) * s[2]] = True
            overlay = overlay.reshape([s[0], s[1], s[3], s[4], s[2]])
            return np.rollaxis(overlay, -1, 2)            
        
    
    def get_reconstructed_input(self, hidden, reduced=False,
                                overlay=None, static_hidden=False, scale=True):
        """
        overlay mask holds positions of max indices (when max pooling was done).
        If None, use previous state where possible.
        If None, and no previous state, assign random positions.
        If scalar, set max indices to this.
        If list, put in multiple positions.
        Same random position is assigned to every hidden
        """        

        if not reduced:
            return Conv.get_reconstructed_input(self, hidden)

        hidden = tf.tile(tf.expand_dims(hidden, 3),
                         [1, 1, self.pool_width, 1, 1])

        if overlay is None:
            overlay = self.state.get('overlay')
            if overlay is None:
                overlay = self._random_overlay(static_hidden=static_hidden)
        elif np.isscalar(overlay) or type(overlay) == list:
            if scale and type(overlay) == list and len(overlay) > 1:
                scale = 1. / len(overlay)
            else: scale = None
            overlay = self._make_overlay(overlay)
            
        return Conv.get_reconstructed_input(self,
                                  self._pool_overlay(hidden, overlay), scale=scale)


class RBM(Auto):
    """Restricted Boltzmann Machine. Adapted from deeplearning.net."""

    attr_names = Auto.attr_names + ['CDk', 'persistent']

    def __init__(self, CDk=2, persistent=False, **kwargs):
        """
        CDk:
        Number of Gibbs sampling steps to use for contrastive divergence
        when computing cost.

        persistent:
        Set to True to use persistent CD.
        """

        Auto.__init__(self, **kwargs)
        if not kwargs.get('fromfile'):
            self.CDk = CDk 
            self.persistent = persistent

        # Store end of the Gibbs chain here.
        self.chain_end = None
        

    def init_train_args(self, **kwargs):
        self.train_args = [tf.placeholder(self.dtype,
                                          shape=[None, self.n_visible]),
                           tf.placeholder(self.dtype,
                                          shape=[None, self.n_visible])]
        self.input_dims = 2


    def free_energy(self, v):
        """Approx free energy of system given visible unit values."""

        Wx_b = tf.matmul(v, self.params['W']) + self.params['bhid']
        vbias_term = tf.squeeze(tf.matmul(v, tf.expand_dims(self.params['bvis'],
                                                            1)))
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(Wx_b)),
                                    reduction_indices=[1])

        return -hidden_term - vbias_term
    
    
    def grad_free_energy(self, inputs):
        """
        Return gradient of free energy w.r.t. each bit of sample.
        """
        wx_b = tf.matmul(inputs, self.params['W']) + self.params['bhid']
        return -tf.matmul(tf.sigmoid(wx_b), self.params['W'], transpose_b=True)\
               - self.params['bvis']


    def energy(self, v, h):
        """Energy of system given visible unit and hidden values."""

        vbias_term = tf.matmul(v, tf.expand_dims(self.params['bvis'], 1))
        hidden_term = tf.matmul(h, tf.expand_dims(self.params['bhid'], 1))
        Wx = tf.matmul(v, self.params['W'])
        hWx = tf.batch_matmul(tf.expand_dims(Wx, 1), tf.expand_dims(h, 2))

        return -tf.squeeze(hidden_term) - tf.squeeze(vbias_term) - tf.squeeze(hWx)
    
        
    def grad_energy_v(self, h):
        """Gradient of energy with respect to visible values."""
        return -tf.expand_dims(self.params['bvis'], 0) \
               -tf.matmul(h, self.params['W'], transpose_b=True)


    def grad_energy_h(self, v):
        """Gradient of energy with respect to hidden values."""
        return -tf.expand_dims(self.params['bhid'], 0) \
               -tf.matmul(v, self.params['W'])

        
    def sample_h_given_v(self, v):
        """
        Given visible unit values, sample hidden unit values.

        Note: implemented in numpy for efficiency.
              Do not use in computation graph.
        """

        mean_h = sigmoid(np.dot(v, self.params['W'].eval()) + self.params['bhid'].eval())
        rnds = random_sample(mean_h.shape)
        return (mean_h > rnds).astype(np.float32)
        

    def sample_v_given_h(self, h):
        """
        Given hidden unit values, sample visible unit values.
        
        Note: implemented in numpy for efficiency.
              Do not use in computation graph.
        """

        mean_v = sigmoid(np.dot(h, self.params['W'].eval().T) +
                         self.params['bvis'].eval())
        rnds = random_sample(mean_v.shape)
        return (mean_v > rnds).astype(np.float32)


    def gibbs_hvh(self, h0):
        """
        This function implements one step of Gibbs sampling,
        starting from the hidden state.
        """
        v1 = self.sample_v_given_h(h0)
        h1 = self.sample_h_given_v(v1)
        return v1, h1


    def gibbs_vhv(self, v0):
        """
        This function implements one step of Gibbs sampling,
        starting from the visible state.
        """
        h1 = self.sample_h_given_v(v0)
        v1 = self.sample_v_given_h(h1)
        return h1, v1


    def sample_chain(self, inputs):
        """
        Start or resume (if self.persistent) Gibbs sampling.

        Return sample at end of the chain.
        """
        
        if not self.persistent or self.chain_end is None:
            self.chain_end = [None, self.sample_h_given_v(inputs)]
        elif len(self.chain_end[1]) != len(inputs):
            if self.verbose:
                print('Resetting chain, with new no. of parallel chains: %d'
                      % len(inputs))
            self.chain_end = [None, self.sample_h_given_v(inputs)]

        for k in range(self.CDk):
            self.chain_end = list(self.gibbs_hvh(self.chain_end[1]))

        return self.chain_end
    

    def cost(self, inputs, chain_sample, **kwargs):
        """
        Cost for given input batch of samples, under current params.
        Using free energy and contrastive divergence.
        """

        return tf.reduce_mean(self.free_energy(inputs) - 
                              self.free_energy(chain_sample))


    def np_reconstructed_input(self, hidden):
        """
        Note: implemented in numpy for efficiency.
              Do not use in computation graph.

        Given hidden unit values, return expected visible unit values.
        """
        W = self.params['W'].eval().T
        bvis = self.params['bvis'].eval()
        return sigmoid(np.dot(hidden, W) + bvis)


    def get_samples(self, initial, count=1, burn=1000, step=100):
        """
        Get Gibbs samples from the RBM, as a list with <count> elements.

        initial:
        Seed chain with this input sample, which could be a batch.

        count:
        No. of samples to return. If <initial> is a batch,
        batch number of samples are returned for every <count>.


        burn:
        Burn these many entries from the start of the Gibbs chain.

        step:
        After the first sample, take these many Gibbs steps between
        subsequent samples to be returned.
        """
        
        if count < 1: return []
        
        chain_end = [None, self.sample_h_given_v(initial)]
        for i in range(burn):
            chain_end = self.gibbs_hvh(chain_end[1])
            
        results = [self.np_reconstructed_input(chain_end[1])]
        for i in range(count - 1):
            for j in range(step):
                chain_end = self.gibbs_hvh(chain_end[1])
            results.append(self.np_reconstructed_input(chain_end[1]))

        return results


    def train_feed(self, data):
        """Return feed_dict based on data to be used for training."""

        return {self.train_args[0]: data,
                self.train_args[1]: self.sample_chain(data)[0]}
    
            
    def cost_args(self, data):
        """Return args to self.cost() for given dataset."""

        if not self.persistent or self.chain_end is None:
            return [data, self.sample_chain(data)[0]]

        # use existing end of persistent chain
        return [data, self.chain_end[0]]


class RRBM(RBM):
    """Real valued Restricted Boltzmann Machine."""

    attr_names = RBM.attr_names + ['sigma', 'beta_sampling']

    def __init__(self, sigma=0.1, beta_sampling=True, **kwargs):
        """
        sigma:
        Standard deviation of input data, for use in sampling.

        beta_sampling:
        Use beta distribution for sampling, instead of Gaussian.
        """
        
        RBM.__init__(self, **kwargs)
        if not kwargs.get('fromfile'):
            self.sigma = sigma
            self.beta_sampling = beta_sampling
        if self.sigma is None: raise AssertionError('Need to supply sigma param.')


    def free_energy(self, v):
        return RBM.free_energy(self, v) + \
               .5 * tf.reduce_sum((v - .5)**2, reduction_indices=[1])


    def sample_v_given_h(self, h, eps=1e-5):
        
        mean_v = sigmoid(np.dot(h, self.params['W'].eval().T) +
                         self.params['bvis'].eval())

        if not self.beta_sampling:
            rnds = np.random.randn(mean_v.shape[0], mean_v.shape[1]).astype(h.dtype)
            return np.clip(mean_v + rnds * self.sigma, 0, 1)
        
        mvvm = mean_v * (1 - mean_v)
        var_v = np.fmin(mvvm, self.sigma**2)
        operand = (mvvm + 1.5 * eps) / (var_v + eps) - 1
        alpha = mean_v * operand
        beta = (1 - mean_v) * operand + eps

        return np.random.beta(alpha, beta).astype(h.dtype)


class R3BM(RRBM):
    """
    Real-valued RBM with real-valued hidden units too.
    """
    
    def sample_h_given_v(self, v, eps=1e-5):
        
        mean_h = sigmoid(np.dot(v, self.params['W'].eval()) +
                         self.params['bhid'].eval())

        if not self.beta_sampling:
            rnds = np.random.randn(mean_h.shape[0], mean_h.shape[1]).astype(v.dtype)
            return np.clip(mean_h + rnds * self.sigma, 0, 1)
        
        mhhm = mean_h * (1 - mean_h)
        var_h = np.fmin(mhhm, self.sigma**2)
        operand = (mhhm + 1.5 * eps) / (var_h + eps) - 1
        alpha = mean_h * operand
        beta = (1 - mean_h) * operand + eps

        return np.random.beta(alpha, beta).astype(v.dtype)


class ERBM(RBM):
    """
    RBM trained given both visible and hidden values.

    We maximize log likelihood by changing the params without
    inferring the hidden values, just taking them as given instead.
    """

    def init_train_args(self, **kwargs):
        """
        Two sets: the given input data, and the sample from the Markov chain.
        Each set has visible values and hidden values.
        
        TensorFlow doesn't like compound tuples/lists in the feed dict,
        so flatten the two sets into a simple list.
        """
        self.train_args = \
            [tf.placeholder(self.dtype, shape=[None, self.n_visible]),
             tf.placeholder(self.dtype, shape=[None, self.n_hidden]),
             tf.placeholder(self.dtype, shape=[None, self.n_visible]),
             tf.placeholder(self.dtype, shape=[None, self.n_hidden])]
        self.input_dims = 2


    def cost(self, v, h, chain_v, chain_h):
        """
        Cost for given input batch of samples, under current params.
        Using energy and contrastive divergence.
        """

        return tf.reduce_mean(self.energy(v, h) - 
                              self.energy(chain_v, chain_h))


    def data_split(self, data):
        """
        Training data comprises visible and hidden values in single rows.
        Split thess rows into the two sets.
        """
        return data[:, :self.n_visible], data[:, self.n_visible:]

    
    def cost_args(self, data):

        v, h = self.data_split(data)

        if not self.persistent or self.chain_end is None:
            return [v, h] + self.sample_chain(v)

        # use existing end of persistent chain
        return [v, h] +  self.chain_end


    def train_feed(self, data):

        states, outputs = self.data_split(data)
        chain_end = self.sample_chain(states)
        
        return {self.train_args[0]: states,
                self.train_args[1]: outputs,
                self.train_args[2]: chain_end[0],
                self.train_args[3]: chain_end[1]}


    def rms_loss(self, data):
        """
        Root-mean-squared difference between <inputs> and encoded-decoded output.
        """
        v, h = self.data_split(data)
        RBM.rms_loss(self, v)



class RNN(Coder):
    """Recurrent neural network."""

    param_names = ['Wxh', 'Whh', 'Why', 'bhid', 'bvis']

    @staticmethod
    def prep_mnist(data):
        """Reshape MNIST data from batch x pixels to batch x columns x rows."""
        return data.reshape((-1, 28, 28)).swapaxes(1, 2)

    
    def __init__(self, n_visible=28, n_hidden=100, seq_length=28, **kwargs):
        """
        seq_length is length of sequences used for training.
        """
                     
        self.seq_length = seq_length
        Coder.__init__(self, n_visible=n_visible, n_hidden=n_hidden, **kwargs)

    def init_params(self, trainable=True):

        self.params['Wxh'] = xavier_init(self.n_visible, self.n_hidden,
                                         name='Wxh', trainable=trainable, dtype=self.dtype)
        self.params['Whh'] = xavier_init(self.n_hidden, self.n_hidden,
                                         name='Whh', trainable=trainable, dtype=self.dtype)

        if 'Why' in type(self).param_names:
            self.params['Why'] = xavier_init(self.n_hidden, self.n_visible,
                                             name='Why', trainable=trainable, dtype=self.dtype)

        self.params['bhid'] = tf.Variable(tf.zeros([self.n_hidden], dtype=self.dtype),
                                          name='bhid', trainable=trainable)
        self.params['bvis'] = tf.Variable(tf.zeros([self.n_visible], dtype=self.dtype),
                                          name='bvis', trainable=trainable)


    def init_train_args(self, **kwargs):
        # To be used for training by tf.Optimizer objects.
        self.train_args = [tf.placeholder(self.dtype,
                                          shape=[None, self.n_visible, self.seq_length])]
        self.input_dims = 2


    def update(self, states, inputs):
        """
        Update states given old states and new inputs.
        Return new states and outputs corresponding to inputs.
        """

        operand = tf.matmul(inputs, self.params['Wxh']) + self.params['bhid']
        if states is not None: operand += tf.matmul(states, self.params['Whh'])
        states = self.coding(operand)
        outputs = self.coding(tf.matmul(states,
                                        self.params.get('Why',
                                             tf.transpose(self.params['Wxh']))) +
                              self.params['bvis'])

        return states, outputs
        

    def get_outputs(self, inputs, state, outputs=[], states=[], as_list=False):
        """
        Computes the output values of the RNN for each of input,
        assuming inputs is a batch.
        
        i.e. inputs is batch_size x n_visible x seq_size.

        Returns the outputs and states.
        """

        for row in tf.unpack(tf.transpose(inputs, perm=[2, 0, 1])):
            state, output = self.update(state, row)
            states.append(state)
            outputs.append(output)

        if not as_list:
            outputs = tf.transpose(tf.pack(outputs), perm=[1, 2, 0])
            states = tf.transpose(tf.pack(states), perm=[1, 2, 0])
        return outputs, states


    def recode(self, inputs, **kwargs):
        """
        Wrapper around get_outputs() to fit recode() signature of parent class.
        """
        outputs, states = \
           self.get_outputs(inputs, None, outputs=[inputs[:, :, 0]],
                            states=[], as_list=True)
        return tf.transpose(tf.pack(outputs[:-1]), perm=[1, 2, 0])


    def get_hidden_values(self, visible, states=None,
                          precoding=False, as_list=False, **kwargs):
        """
        Given visible unit values, return expected hidden unit values.
        """

        hidden = []
        for row in tf.unpack(tf.transpose(visible, perm=[2, 0, 1])):
            operand = tf.matmul(row, self.params['Wxh']) + self.params['bhid']
            if states is not None:
                operand += tf.matmul(states, self.params['Whh'])
            hidden.append(operand if precoding else self.coding(operand))
            states = self.coding(operand) if precoding else hidden[-1]

        if as_list: return hidden
        
        return tf.transpose(tf.pack(hidden), perm=[1, 2, 0])


    def extend(self, inputs, length, as_list=False):
        """
        Given initial sequence of inputs, extend by given length, feeding
        RNN outputs back to itself.

        Return full output and ending state.
        """

        outputs, states = \
           self.get_outputs(inputs, None, [inputs[:, :, 0]], as_list=True)

        state = states[-1]
        for index in range(1, length):
            state, output = self.update(state, outputs[-1])
            outputs.append(output)
            states.append(state)
            
        if not as_list:
            outputs = tf.transpose(tf.pack(outputs), perm=[1, 2, 0])
            states = tf.transpose(tf.pack(states), perm=[1, 2, 0])
        return outputs, states


    def features(self, sample, length=None, scale=True):
        """
        Returns a rasterized 2D image for each hidden unit, along with
        the shape of each (unrasterized) such image.

        Top half of the image comprises the <length> rows of input that tend to
        activate the hidden unit.

        This is computed by sampling <length> sequential rows from <sample>
        input, computing end states when the RNN is fed these sequences, and
        taking a weighted average of the sequential rows where the weights are
        the pre-sigmoid state values (floored at 0.0).

        Bottom half of the image comprises <length> sequential rows of output
        when the RNN is rolled forward from the end-point of the top half.

        This is computed by taking a weighted average of the sequential rows of
        rolled forward output where the weights are the same weights as for the
        top half computation.

        Middle two rows of the image comprise Wxh and Why.T weights.

        scale: Normalize features for each hidden unit to between 0.0 amd 1.0.
        """

        row_len = self.n_visible
        if length is None: length = row_len / 2

        #### sample sequential rows from <sample> ####
        times = row_len // length
        sample_size = len(sample) * times
        inputs = []
        for s in sample:
            ends = randint(length, row_len, times)
            for e in ends: inputs.append(s.T[:e])

        # We switch to numpy for processing, as it is much faster when
        # we run the RNN one row at a time one sample at a time.
        Wxh = self.params['Wxh'].eval()
        Whh = self.params['Whh'].eval()
        Why = self.params['Why'].eval() if 'Why' in self.params else Wxh.T
        bhid = self.params['bhid'].eval()
        bvis = self.params['bvis'].eval()
        
        states, outputs = [], []
        for sample in inputs:
            # For top half of feature image.
            s = np.zeros((1, self.n_hidden), dtype=np.float32)
            for row in sample:
                s = sigmoid(np.dot(row, Wxh) + np.dot(s, Whh) + bhid)
            states.append(np.fmax(logit(s), 0))
            
            # For bottom half of feature image.
            o = [sigmoid(np.dot(s, Why) + bvis)]
            for i in range(length - 1):
                s = sigmoid(np.dot(o[-1], Wxh) + np.dot(s, Whh) + bhid)
                o.append(sigmoid(np.dot(s, Why) + bvis))
            outputs.append(np.array(o).squeeze().flatten())

        weights = np.array(states).squeeze()

        # top half
        rows = np.array([i[-length:].flatten() for i in inputs])
        prefix = np.dot(rows.T, weights) / np.sum(weights, axis=0)
        # bottom half
        suffix = np.dot(np.array(outputs).T, weights) / np.sum(weights, axis=0)
        
        if scale:
            prefix = unit_scale(prefix, axis=0)
            suffix = unit_scale(suffix, axis=0)

        results = np.concatenate((prefix,
#                                  np.zeros(Wxh.shape),
                                  unit_scale(Wxh, axis=0),
                                  unit_scale(Why.T, axis=0),
                                  suffix))
        return results.T, (2 * length + 2, row_len)


class RNNtie(RNN):
    """Recurrent neural network with tied weights."""
    param_names = ['Wxh', 'Whh', 'bhid', 'bvis']



class VAE(Coder):
    """Variational Autoencoder."""
    
    param_names = ['Wxh', 'Mhz', 'Shz', 'Wzh', 'Mhx', 'Shx',
                   'bWxh', 'bMhz', 'bShz', 'bWzh', 'bMhx', 'bShx']
    attr_names = Coder.attr_names + ['n_z']

    def __init__(self, n_z=20, **kwargs):
        """
        n_z:
        Dimension of latent variables.
        """
        
        self.n_z = n_z
        Coder.__init__(self, **kwargs)

        # Train args has an additional element: sampling from latent space
        self.train_args.append(tf.placeholder(float_dt,
                                              shape=[None, self.n_z]))


    def init_params(self, constant=1, trainable=True, **kwargs):

        self.params['Wxh'] = xavier_init(self.n_visible, self.n_hidden,
                                         'Wxh', constant, trainable=trainable)
        self.params['Mhz'] = xavier_init(self.n_hidden, self.n_z,
                                         'Mhz', constant, trainable=trainable)
        self.params['Shz'] = xavier_init(self.n_hidden, self.n_z,
                                         'Shz', constant, trainable=trainable)

        self.params['Wzh'] = xavier_init(self.n_z, self.n_hidden,
                                         'Wzh', constant, trainable=trainable)
        self.params['Mhx'] = xavier_init(self.n_hidden, self.n_visible,
                                         'Mhx', constant, trainable=trainable)
        self.params['Shx'] = xavier_init(self.n_hidden, self.n_visible,
                                         'Shx', constant, trainable=trainable)

        self.params['bWxh'] = tf.Variable(tf.zeros([self.n_hidden], dtype=float_dt), name='bWxh', trainable=trainable)
        self.params['bMhz'] = tf.Variable(tf.zeros([self.n_z], dtype=float_dt), name='bMhz', trainable=trainable)
        self.params['bShz'] = tf.Variable(tf.zeros([self.n_z], dtype=float_dt), name='bShz', trainable=trainable)

        self.params['bWzh'] = tf.Variable(tf.zeros([self.n_hidden], dtype=float_dt), name='bWxh', trainable=trainable)
        self.params['bMhx'] = tf.Variable(tf.zeros([self.n_visible], dtype=float_dt), name='bMhx', trainable=trainable)
        self.params['bShx'] = tf.Variable(tf.zeros([self.n_visible], dtype=float_dt), name='bMhx', trainable=trainable)


    def get_h_inputs(self, inputs):
        """Computes the values of the hidden layer, given inputs."""
        return self.coding(tf.matmul(inputs, self.params['Wxh']) +
                           self.params['bWxh'])
    

    def get_hidden_values(self, inputs, **kwargs):
        """Computes the values of the latent z variables."""
        h = self.get_h_inputs(inputs)
        return tf.matmul(h, self.params['Mhz']) + self.params['bMhz']


    def get_h_latents(self, latents):
        """Computes the values of the hidden layer, generative network."""
        return self.coding(tf.matmul(latents, self.params['Wzh']) +
                          self.params['bWzh'])


    def get_reconstructed_input(self, latents, **kwargs):
        """
        Computes the reconstructed input given the values of the
        latent z variables.
        """
        h = self.get_h_latents(latents)
        return self.decoding(tf.matmul(h, self.params['Mhx']) +
                             self.params['bMhx'])


    def cost(self, inputs, variation, eps=1e-5, **kwargs):
        """
        Cost for given input batch of samples, under current params.
        """
        h = self.get_h_inputs(inputs)
        z_mu = tf.matmul(h, self.params['Mhz']) + self.params['bMhz']
        z_sig = tf.matmul(h, self.params['Shz']) + self.params['bShz']

        # KL divergence between latent space induced by encoder and ...
        lat_loss = -tf.reduce_sum(1 + z_sig - z_mu**2 - tf.exp(z_sig), 1)

        z = z_mu + tf.sqrt(tf.exp(z_sig)) * variation
        h = self.get_h_latents(z)
        x_mu = self.decoding(tf.matmul(h, self.params['Mhx']) + self.params['bMhx'])
        x_sig = self.decoding(tf.matmul(h, self.params['Shx']) + self.params['bShx'])
#        x_sig = tf.clip_by_value(x_mu * (1 - x_mu), .05, 1)

        # decoding likelihood term
        like_loss = tf.reduce_sum(tf.log(x_sig + eps) +
                                  (inputs - x_mu)**2 / x_sig, 1)

#        # Mean cross entropy between input and encode-decoded input.
#        like_loss = 2 * tf.reduce_sum(functions.cross_entropy(inputs, x_mu), 1)
        
        return .5 * tf.reduce_mean(like_loss + lat_loss)


    def train_feed(self, data):
        """Return feed_dict based on data to be used for training."""

        return {self.train_args[0]: data,
                self.train_args[1]: randn(data.shape[0], self.n_z)}


    def cost_args(self, data):
        """Return args to self.cost() for given dataset."""

        return [data, randn(data.shape[0], self.n_z)]


    def features(self, *args):
        """Return weights in suitable manner for plotting."""
        return self.params['Wxh'].eval().T

