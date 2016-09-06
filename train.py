"""
Utilities for training networks.* objects.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import division
#from __future__ import print_function

import numpy as np
from numpy.random import randint

import tensorflow as tf

def corrupt(dataset, corruption):
    """
    Return a corrupted copy of the input dataset.
    
    corruption: (between 0.0 and 1.0)
    Fraction of dataset randomly set to mean value of the dataset.
    """
    
    corrupted = dataset.flatten()
    size = corrupted.size
    
    corrupted[randint(size,
                      size=int(corruption * size))] = np.mean(corrupted)
    
    return corrupted.reshape(dataset.shape)


def get_costs(coder, dataset, batch_size=100):
    """
    Return average cost function value and rms-loss value on validation
    set by coder object with its current weights.
    """
    
    n_batches = dataset.shape[0] // batch_size
    cost, rms_loss = 0, 0
    for index in range(n_batches):
        batch = dataset[index * batch_size : (index+1) * batch_size]
        cost += coder.cost(*coder.cost_args(batch)).eval()
        rms_loss += coder.rms_loss(batch).eval()

    return (cost / n_batches, rms_loss / n_batches)


def train(sess, coder, dataset, validation_set, verbose=False,
          training_epochs=10, learning_rate=0.001, batch_size=100,
          corruption=None):
    """
    Train a networks object on given data.

    sess: TensorFlow session.

    coder: networks object that supports cost() and rms_loss() methods.

    dataset: dataset for training.

    validation_set: dataset for monitoring.
    """

    train_step = tf.train.AdamOptimizer(learning_rate)\
                     .minimize(coder.cost(*coder.train_args))
    sess.run(tf.initialize_all_variables())

    if verbose: print('Initial cost %.2f, r.m.s. loss %.3f' %
                      get_costs(coder, validation_set, batch_size))
    
    n_train_batches = dataset.shape[0] // batch_size
    for epoch in range(training_epochs):

        for index in range(n_train_batches):
            batch = dataset[index * batch_size : (index+1) * batch_size]
            train_step.run(feed_dict=coder.train_feed(batch))

        if verbose: print('Training epoch %d, cost %.2f, r.m.s. loss %.3f ' %
                          ((epoch,) + get_costs(coder, validation_set, batch_size)))


