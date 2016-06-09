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
from numpy.random import randint, random_sample
import cPickle

import functions
from tamarind.functions import sigmoid, unit_scale

class Auto(object):
    """Auto-Encoder. Adapted from deeplearning.net."""

    def __init__(self, n_visible=784, n_hidden=500, corruption=0.0,
                 random_seed=123, fromfile=None, W=None, bhid=None, bvis=None):
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

        :type fromfile: str
        :param fromfile:  initialize params from this saved file

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
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.corruption = corruption

        if fromfile is not None:
            save_file = open(fromfile)
            W = tf.Variable(cPickle.load(save_file), name='W')
            bhid = tf.Variable(cPickle.load(save_file), name='bhid')
            bvis = tf.Variable(cPickle.load(save_file), name='bvis')
            save_file.close()
            
        if not W:
            # Uniformly sample from +/-4*sqrt(6./(n_visible+n_hidden)).
            if random_seed is not None: tf.set_random_seed(random_seed)
            Winit_max = 4 * np.sqrt(6. / (n_hidden + n_visible))
            W = tf.Variable(tf.random_uniform([n_visible, n_hidden],
                                              minval=-Winit_max, maxval=Winit_max,
                                              dtype=tf.float32), name='W')

        if not bvis: bvis = tf.Variable(tf.zeros([n_visible]), name='bvis')
        if not bhid: bhid = tf.Variable(tf.zeros([n_hidden]), name='bhid')

        self.W = W
        self.bhid = bhid
        self.bvis = bvis
        self.params = [self.W, self.bhid, self.bvis]

    
    def save_params(self, filename):
        """
        Save params to disk, compatible with fromfile option of constructor.
        """
        save_file = open(filename, 'wb')
        for variable in self.params:
            cPickle.dump(variable.eval(), save_file, -1) 
        save_file.close()    
        
        
    def get_hidden_values(self, inputs):
        """Computes the values of the hidden layer."""
        return tf.sigmoid(tf.matmul(inputs, self.W) + self.bhid)


    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the
        hidden layer.
        """
        return tf.sigmoid(tf.matmul(hidden, self.W, transpose_b=True) +
                          self.bvis)


    def recode(self, inputs):
        """Encode and then decode input, using current params."""
        y = self.get_hidden_values(inputs)
        return self.get_reconstructed_input(y)


    def cost(self, inputs, target=None):
        """
        Cost for given input batch of samples, under current params.
        Mean cross entropy.
        """
        if target is None: target = inputs
        loss = functions.cross_entropy(target, self.recode(inputs))
        return tf.reduce_mean(-tf.reduce_sum(loss, reduction_indices=[1]))


    def rms_loss(self, inputs, target=None):
        """
        Root-mean-squared difference between <inputs> and encoded-decoded output.
        """
        loss = (inputs - self.recode(inputs)) ** 2
        return tf.reduce_mean(tf.reduce_mean(loss, reduction_indices=[1]) ** .5)
        

    def features(self, *args):
        """Return weights in suitable manner for plotting."""
        return self.W.eval().T


class RBM(Auto):
    """Restricted Boltzmann Machine. Adapted from deeplearning.net."""

    def __init__(self, CDk=15, **kwargs):
        """
        CDk:
        Number of Gibbs sampling steps to use for contrastive divergence
        when computing cost.
        """
        
        Auto.__init__(self, **kwargs)
        self.CDk = CDk
        

    def free_energy(self, v):
        """Approx free energy of system given visible unit values."""
#        Wx_b = tf.matmul(v, self.W) + self.bhid
#        vbias_term = tf.matmul(v, tf.expand_dims(self.bvis, 1))
#        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(Wx_b)),
#                                    reduction_indices=[1])
#        return -hidden_term - vbias_term

        product = tf.reduce_sum(v * self.recode(v), reduction_indices=[1])
        return -tf.log(product)
    
    
    def sample_h_given_v(self, v):
        """Given visible unit values, sample hidden unit values."""
        mean_h = self.get_hidden_values(v)
        rnds = random_sample(mean_h.get_shape().as_list())
        return tf.cast(mean_h > rnds, tf.float32)
        

    def sample_v_given_h(self, h):
        """Given hidden unit values, sample visible unit values."""
        mean_v = self.get_reconstructed_input(h)
        rnds = random_sample(mean_v.get_shape().as_list())
        return tf.cast(mean_v > rnds, tf.float32)


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


    def cost(self, inputs):
        """
        Cost for given input batch of samples, under current params.
        Mean cross entropy.
        """

        # XXX considered using tf.stop_gradient() but it turned out to
        #     be unnecessary.
        chain_end = None, self.sample_h_given_v(inputs)
        for k in range(self.CDk):
            chain_end = self.gibbs_hvh(chain_end[1])

        sample = chain_end[0]
        return tf.reduce_mean(self.free_energy(inputs)) - \
               tf.reduce_mean(self.free_energy(sample))


class RNN(Auto):
    """Recurrent neural network."""

    def __init__(self, n_visible=28, n_hidden=100, tie_Wxy=True,
                 random_seed=123, fromfile=None, params=None,
                 Wxh=None, Whh=None, Why=None, bhid=None, bvis=None):
        """
        params is a list of Wxh, Whh, Why, bhid and bvis in order.
        """
                     
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if fromfile is not None:
            save_file = open(fromfile)
            Wxh = tf.Variable(cPickle.load(save_file), name='Wxh')
            Whh = tf.Variable(cPickle.load(save_file), name='Whh')
            if tie_Wxy: Why = None
            else: Why = tf.Variable(cPickle.load(save_file), name='Why')
            bhid = tf.Variable(cPickle.load(save_file), name='bhid')
            bvis = tf.Variable(cPickle.load(save_file), name='bvis')
            save_file.close()

        elif params is not None:
            Wxh, Whh, Why, bhid, bvis = params

        else:
            # Uniformly sample from +/-4*sqrt(6./(n_visible+n_hidden)).
            if random_seed is not None: tf.set_random_seed(random_seed)
            Winit_max = 4 * np.sqrt(6. / (n_hidden + n_visible))
            Wxh = tf.Variable(tf.random_uniform([n_visible, n_hidden],
                                                minval=-Winit_max, maxval=Winit_max,
                                                dtype=tf.float32), name='Wxh')
            if tie_Wxy: Why = None
            else:
                Why = tf.Variable(tf.random_uniform([n_hidden, n_visible],
                                                    minval=-Winit_max, maxval=Winit_max,
                                                    dtype=tf.float32), name='Wxh')
            Winit_max = 4 * np.sqrt(3. / n_hidden)
            Whh = tf.Variable(tf.random_uniform([n_hidden, n_hidden],
                                                minval=-Winit_max, maxval=Winit_max,
                                                dtype=tf.float32), name='Wxh')
            bvis = tf.Variable(tf.zeros([n_visible]), name='bvis')
            bhid = tf.Variable(tf.zeros([n_hidden]), name='bhid')

        self.Wxh = Wxh
        self.Whh = Whh
        self.Why = Why
        self.bhid = bhid
        self.bvis = bvis
        if tie_Wxy:
            self.params = [self.Wxh, self.Whh, self.bhid, self.bvis]
        else:
            self.params = [self.Wxh, self.Whh, self.Why, self.bhid, self.bvis]


    def update(self, states, inputs):
        """
        Update states given old states and new inputs.
        Return new states and outputs corresponding to inputs.
        """
        
        states = tf.sigmoid(tf.matmul(inputs, self.Wxh) +
                            tf.matmul(states, self.Whh) +
                            self.bhid)
        outputs = tf.sigmoid(tf.matmul(states, self.Why or tf.transpose(self.Wxh)) +
                             self.bvis)

        return states, outputs
        

    def get_outputs(self, inputs, states, outputs=[],
                    sample_size=None, as_list=False):
        """
        Computes the output values of the RNN for each of input, assuming
        inputs is a batch. i.e. inputs is batch_size x (n_visible * seq_size).
        """

        if sample_size is None:
            sample_size = inputs.get_shape().as_list()[1]
            
        for index in range(0, sample_size, self.n_visible):
            states, output = self.update(states,
                                         inputs[:, index:index+self.n_visible])
            outputs.append(output)

        if not as_list: outputs = tf.concat(1, outputs, name='outputs')
        return outputs, states


    def recode(self, inputs):
        """
        Wrapper around get_outputs() to fit recode() signature of parent class.
        """

        if type(inputs) == np.ndarray:
            inputs = tf.constant(inputs, dtype=tf.float32)
        batch_size, sample_size = inputs.get_shape().as_list()
        
        outputs, states = \
           self.get_outputs(inputs[:, :sample_size-self.n_visible],
                            tf.zeros([batch_size, self.n_hidden], dtype=tf.float32),
                            [inputs[:, :self.n_visible]],
                            sample_size=sample_size-self.n_visible)
        return outputs


    def extend(self, inputs, length, as_list=False):
        """
        Given initial sequence of inputs, extend by given length, feeding
        RNN outputs back to itself.

        Return full output and ending state.
        """
        
        if type(inputs) == np.ndarray:
            inputs = tf.constant(inputs, dtype=tf.float32)
        batch_size, sample_size = inputs.get_shape().as_list()
        
        outputs, states = \
           self.get_outputs(inputs,
                            tf.zeros([batch_size, self.n_hidden], dtype=tf.float32),
                            outputs=[inputs[:, :self.n_visible]],
                            sample_size=sample_size, as_list=True)

        for index in range(1, length):
            states, output = self.update(states, outputs[-1])
            outputs.append(output)
            
        if not as_list: outputs = tf.concat(1, outputs, name='outputs')
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
            for e in ends: inputs.append(s[:e * row_len])

        Wxh = self.Wxh.eval()
        Whh = self.Whh.eval()
        Why = self.Why.eval() if self.Why else Wxh.T
        bhid = self.bhid.eval()
        bvis = self.bvis.eval()
        states, outputs = [], []
        for sample in inputs:
            s = np.zeros((1, self.n_hidden), dtype=np.float32)
            for index in range(0, len(sample), row_len):
                s = sigmoid(\
                    np.dot(sample[index : index + row_len], Wxh) +
                    np.dot(s, Whh) + bhid)
            states.append(s)
            
            o = [sigmoid(np.dot(s, Why) + bvis)]
            for i in range(length - 1):
                s = sigmoid(np.dot(o[-1], Wxh) + np.dot(s, Whh) + bhid)
                o.append(sigmoid(np.dot(s, Why) + bvis))
            outputs.append(np.array(o).squeeze().flatten())

        weights = np.array(states).squeeze()**2
        rows = np.array([i[-length * row_len:] for i in inputs])

        prefix = np.dot(rows.T, weights) / np.sum(weights, axis=0)
        suffix = np.dot(np.array(outputs).T, weights) / np.sum(weights, axis=0)
        results = np.concatenate((prefix, prefix[:row_len], suffix))
        if scale: results = unit_scale(results, axis=0)

        results[len(prefix):len(prefix) + row_len] = unit_scale(Why.T, axis=0)
        
        return results.T, (2 * length + 1, row_len)

