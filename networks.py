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
import numpy as np
from numpy.random import randint, random_sample, randn
import cPickle
from collections import OrderedDict

import functions, train
from tamarind.functions import sigmoid, unit_scale
from scipy.stats import multivariate_normal

def xavier_init(fan_in, fan_out, name='W', constant=4): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    bound = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform((fan_in, fan_out),
                                         minval=-bound, maxval=bound, 
                                         dtype=tf.float32, name=name))


class Coder(object):
    """Virtual NN with skeleton implementation of common functions."""

    param_names = []

    def __init__(self, n_visible=784, n_hidden=500, verbose=False,
                 random_seed=123, params=None, fromfile=None, **kwargs):
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

        :type fromfile: str
        :param fromfile:  initialize params from this saved file
        """
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.verbose = verbose
        if random_seed is not None: tf.set_random_seed(random_seed)

        if params is not None:
            self.params = params
        else: 
            self.params = OrderedDict()
            if fromfile is not None:
                save_file = open(fromfile)
                for name in type(self).param_names:
                    self.params[name] = tf.Variable(cPickle.load(save_file),
                                                    name=name)
                save_file.close()
            else:
                self.init_params(**kwargs)
            
        self.init_train_args(**kwargs)

        
    def init_params(self, **kwargs):
        pass

    def init_train_args(self, **kwargs):
        # To be used for training by tf.Optimizer objects.
        self.train_args = [tf.placeholder(tf.float32,
                                          shape=[None, self.n_visible])]

    
    def save_params(self, filename):
        """
        Save params to disk, compatible with fromfile option of constructor.
        """
        save_file = open(filename, 'wb')
        for variable in self.params.values():
            cPickle.dump(variable.eval(), save_file, -1) 
        save_file.close()    
        

    def cost_args(self, dataset):
        """Return args to self.cost() for given dataset."""
        return [dataset]


    def train_feed(self, data):
        """Return feed_dict based on data to be used for training."""
        return {self.train_args[0]: data}


    def get_hidden_values(self, inputs):
        """Computes the values of the hidden layer."""
        return None #implement in subclass
    
        
    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the
        hidden layer.
        """
        return None #implement in subclass


    def recode(self, inputs):
        """Encode and then decode input, using current params."""
        y = self.get_hidden_values(inputs)
        return self.get_reconstructed_input(y)


    def cost(self, inputs):
        """
        Cost for given input batch of samples, under current params.
        Mean cross entropy between input and encode-decoded input.
        """
        loss = functions.cross_entropy(inputs, self.recode(inputs))
        return tf.reduce_mean(-tf.reduce_sum(loss, reduction_indices=[1]))


    def rms_loss(self, inputs):
        """
        Root-mean-squared difference between <inputs> and encoded-decoded output.
        """
        loss = tf.squared_difference(inputs, self.recode(inputs))
        return tf.reduce_mean(tf.reduce_mean(loss, reduction_indices=[1]) ** .5)
        

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


    def init_params(self):

        self.params['W'] = xavier_init(self.n_visible, self.n_hidden, name='W')
        self.params['bhid'] = tf.Variable(tf.zeros([self.n_hidden]), name='bhid')
        self.params['bvis'] = tf.Variable(tf.zeros([self.n_visible]), name='bvis')

        
    def get_hidden_values(self, inputs):
        """Computes the values of the hidden layer."""
        return tf.sigmoid(tf.matmul(inputs, self.params['W']) +
                          self.params['bhid'])


    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the
        hidden layer.
        """
        return tf.sigmoid(tf.matmul(hidden, self.params['W'], transpose_b=True) +
                          self.params['bvis'])


    def features(self, *args):
        """Return weights in suitable manner for plotting."""
        return self.params['W'].eval().T


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
        self.train_args.append(tf.placeholder(tf.float32,
                                              shape=[None, self.n_visible]))
        

    def cost(self, inputs, corrupts):
        """
        Cost for given input batch of samples, under current params.
        Mean cross entropy between input and encode-decoded corrupted
        version of the input.
        """
        loss = functions.cross_entropy(inputs, self.recode(corrupts))
        return tf.reduce_mean(-tf.reduce_sum(loss, reduction_indices=[1]))


    def cost_args(self, data):
        """Return args to self.cost() for given dataset."""
        return [data, train.corrupt(data, self.corruption)]


    def train_feed(self, data):
        """Return feed_dict based on data to be used for training."""
        return {self.train_args[0]: data,
                self.train_args[1]: train.corrupt(data, self.corruption)}



class RBM(Auto):
    """Restricted Boltzmann Machine. Adapted from deeplearning.net."""

    def __init__(self, CDk=1, persistent=False, **kwargs):
        """
        CDk:
        Number of Gibbs sampling steps to use for contrastive divergence
        when computing cost.

        persistent:
        Set to True to use persistend CD.
        """
        
        Auto.__init__(self, **kwargs)
        self.CDk = CDk
        self.persistent = persistent

        # Store end of the Gibbs chain here.
        self.chain_end = None
        

    def init_train_args(self, **kwargs):
        self.train_args = [tf.placeholder(tf.float32,
                                          shape=[None, self.n_visible]),
                           tf.placeholder(tf.float32,
                                          shape=[None, self.n_visible])]


    def free_energy(self, v):
        """Approx free energy of system given visible unit values."""

        Wx_b = tf.matmul(v, self.params['W']) + self.params['bhid']
        vbias_term = tf.squeeze(tf.matmul(v, tf.expand_dims(self.params['bvis'], 1)))
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
    

    def cost(self, inputs, chain_sample):
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
            [tf.placeholder(tf.float32, shape=[None, self.n_visible]),
             tf.placeholder(tf.float32, shape=[None, self.n_hidden]),
             tf.placeholder(tf.float32, shape=[None, self.n_visible]),
             tf.placeholder(tf.float32, shape=[None, self.n_hidden])]


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

    
    def __init__(self, n_visible=28, n_hidden=100, seq_length=28,
                 random_seed=123, fromfile=None, params=None, verbose=False):
        """
        seq_length is length of sequences used for training.
        """
                     
        self.seq_length = seq_length
        Coder.__init__(self, n_visible=n_visible, n_hidden=n_hidden,
                       random_seed=random_seed, fromfile=fromfile,
                       params=params, verbose=verbose)

    def init_params(self):

        self.params['Wxh'] = xavier_init(self.n_visible, self.n_hidden, 'Wxh')
        self.params['Whh'] = xavier_init(self.n_hidden, self.n_hidden, 'Whh')

        if 'Why' in type(self).param_names:
            self.params['Why'] = xavier_init(self.n_hidden, self.n_visible, 'Why')

        self.params['bhid'] = tf.Variable(tf.zeros([self.n_hidden]), name='bhid')
        self.params['bvis'] = tf.Variable(tf.zeros([self.n_visible]), name='bvis')


    def init_train_args(self, **kwargs):
        # To be used for training by tf.Optimizer objects.
        self.train_args = [tf.placeholder(tf.float32,
                                          shape=[None, self.n_visible, self.seq_length])]


    def update(self, states, inputs):
        """
        Update states given old states and new inputs.
        Return new states and outputs corresponding to inputs.
        """

        operand = tf.matmul(inputs, self.params['Wxh']) + self.params['bhid']
        if states is not None: operand += tf.matmul(states, self.params['Whh'])
        states = tf.sigmoid(operand)
        outputs = tf.sigmoid(tf.matmul(states,
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


    def recode(self, inputs):
        """
        Wrapper around get_outputs() to fit recode() signature of parent class.
        """
        outputs, states = \
           self.get_outputs(inputs, None, outputs=[inputs[:, :, 0]],
                            states=[], as_list=True)
        return tf.transpose(tf.pack(outputs[:-1]), perm=[1, 2, 0])


    def get_hidden_values(self, visible, states=None,
                          presigmoid=False, as_list=False):
        """
        Given visible unit values, return expected hidden unit values.
        """

        hidden = []
        for row in tf.unpack(tf.transpose(visible, perm=[2, 0, 1])):
            operand = tf.matmul(row, self.params['Wxh']) + self.params['bhid']
            if states is not None:
                operand += tf.matmul(states, self.params['Whh'])
            hidden.append(operand if presigmoid else tf.sigmoid(operand))
            states = tf.sigmoid(operand) if presigmoid else hidden[-1]

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
        """XXX: Work in progress."""

        row_len = self.n_visible
        if length is None: length = row_len / 2

        times = row_len // length
        sample_size = len(sample) * times
        inputs = []
        for s in sample:
            ends = randint(length, row_len, times)
            for e in ends: inputs.append(s.T[:e])

        Wxh = self.params['Wxh'].eval()
        Whh = self.params['Whh'].eval()
        Why = self.params['Why'].eval() if 'Why' in self.params else Wxh.T
        bhid = self.params['bhid'].eval()
        bvis = self.params['bvis'].eval()
        
        states, outputs = [], []
        for sample in inputs:
            s = np.zeros((1, self.n_hidden), dtype=np.float32)
            for row in sample:
                s = sigmoid(np.dot(row, Wxh) + np.dot(s, Whh) + bhid)
            states.append(s)
            
            o = [sigmoid(np.dot(s, Why) + bvis)]
            for i in range(length - 1):
                s = sigmoid(np.dot(o[-1], Wxh) + np.dot(s, Whh) + bhid)
                o.append(sigmoid(np.dot(s, Why) + bvis))
            outputs.append(np.array(o).squeeze().flatten())

        weights = np.array(states).squeeze()**2
        rows = np.array([i[-length:].flatten() for i in inputs])

        prefix = np.dot(rows.T, weights) / np.sum(weights, axis=0)
        suffix = np.dot(np.array(outputs).T, weights) / np.sum(weights, axis=0)
        results = np.concatenate((prefix, prefix[:row_len], suffix))
        if scale: results = unit_scale(results, axis=0)

        results[len(prefix):len(prefix) + row_len] = unit_scale(Why.T, axis=0)
        
        return results.T, (2 * length + 1, row_len)


class RNNtie(RNN):
    """Recurrent neural network with tied weights."""
    param_names = ['Wxh', 'Whh', 'bhid', 'bvis']



class VAE(Coder):
    """Variational Autoencoder."""
    
    param_names = ['Wxh', 'Mhz', 'Shz', 'Wzh', 'Mhx',
                   'bWxh', 'bMhz', 'bShz', 'bWzh', 'bMhx']

    def __init__(self, n_z=100, **kwargs):
        """
        n_z:
        Dimension of latent variables.
        """
        
        self.n_z = n_z
        Coder.__init__(self, **kwargs)

        # Train args has an additional element: sampling from latent space
        self.train_args.append(tf.placeholder(tf.float32,
                                              shape=[None, self.n_z]))


    def init_params(self, constant=1):

        self.params['Wxh'] = xavier_init(self.n_visible, self.n_hidden,
                                         'Wxh', constant)
        self.params['Mhz'] = xavier_init(self.n_hidden, self.n_z,
                                         'Mhz', constant)
        self.params['Shz'] = xavier_init(self.n_hidden, self.n_z,
                                         'Shz', constant)

        self.params['Wzh'] = xavier_init(self.n_z, self.n_hidden,
                                         'Wzh', constant)
        self.params['Mhx'] = xavier_init(self.n_hidden, self.n_visible,
                                         'Mhx', constant)

        self.params['bWxh'] = tf.Variable(tf.zeros([self.n_hidden]), name='bWxh')
        self.params['bMhz'] = tf.Variable(tf.zeros([self.n_z]), name='bMhz')
        self.params['bShz'] = tf.Variable(tf.zeros([self.n_z]), name='bShz')

        self.params['bWzh'] = tf.Variable(tf.zeros([self.n_hidden]), name='bWxh')
        self.params['bMhx'] = tf.Variable(tf.zeros([self.n_visible]), name='bMhx')


    def get_h_inputs(self, inputs):
        """Computes the values of the hidden layer, given inputs."""
        return tf.sigmoid(tf.matmul(inputs, self.params['Wxh']) +
                          self.params['bWxh'])


    def get_hidden_values(self, inputs):
        """Computes the values of the latent z variables."""
        h = self.get_h_inputs(inputs)
        return tf.matmul(h, self.params['Mhz']) + self.params['bMhz']


    def get_h_latents(self, latents):
        """Computes the values of the hidden layer, generative network."""
        return tf.sigmoid(tf.matmul(latents, self.params['Wzh']) +
                          self.params['bWzh'])


    def get_reconstructed_input(self, latents):
        """
        Computes the reconstructed input given the values of the
        latent z variables.
        """
        h = self.get_h_latents(latents)
        return tf.sigmoid(tf.matmul(h, self.params['Mhx']) +
                          self.params['bMhx'])


    def cost(self, inputs, variation):
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
        x_mu = tf.sigmoid(tf.matmul(h, self.params['Mhx']) + self.params['bMhx'])
        x_sig = tf.clip_by_value(x_mu * (1 - x_mu), .05, 1)

        # decoding likelihood term
        like_loss = tf.reduce_sum(tf.log(x_sig) +
                                  (inputs - x_mu)**2 / x_sig, 1)

#        # Mean cross entropy between input and encode-decoded input.
#        like_loss = -2 * tf.reduce_sum(functions.cross_entropy(inputs, x_mu), 1)
        
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


