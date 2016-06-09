"""
Utilities for training networks.* objects.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import division

import numpy as np
from numpy.random import randint

import tensorflow as tf

def corrupt(dataset, corruption):
    """
    Return a corrupted copy of the input dataset.
    
    corruption: (between 0.0 and 1.0)
    Fraction of dataset randomly set to mean value of the dataset.
    """
    
    corrupted = dataset.copy()
    if not corruption: return corrupted
    
    mean_val = np.mean(corrupted)
    rows, cols = corrupted.shape
    for i in range(rows):
        corrupted[i][randint(cols, size=int(corruption * cols))] = mean_val

    return corrupted


def train(sess, coder, dataset, validation_set, verbose=False,
          training_epochs=10, learning_rate=0.01, batch_size=100,
          corruption=None, **kwargs):
    """
    Train a networks object on given data.

    sess: TensorFlow session.

    coder: networks object that supports cost() and rms_loss() methods.

    dataset: dataset for training.

    validation_set: dataset for monitoring.
    
    corruption:
    Perform adversarial training by corrupting dataset: set this fraction of
    data to the mean value of data, picking the locations randomly each epoch.
    """

    n_visible = dataset.shape[-1]
    n_train_batches = dataset.shape[0] // batch_size
    x = tf.placeholder(tf.float32, shape=[batch_size, n_visible])

    if corruption:
        xc = tf.placeholder(tf.float32, shape=[batch_size, n_visible])
        train_args = [xc, x]
        valid_args = [corrupt(validation_set, corruption), validation_set]
    else: train_args, valid_args = [x], [validation_set]
    
    train_step = tf.train.AdagradOptimizer(learning_rate)\
                     .minimize(coder.cost(*train_args))

    sess.run(tf.initialize_all_variables())
    if verbose:
        print('Initial cost %.2f, r.m.s. loss %.3f' %
              (coder.cost(*valid_args).eval(),
               coder.rms_loss(validation_set).eval()))
    
    for epoch in range(training_epochs):

        for index in range(n_train_batches):
            batch = dataset[index * batch_size:(index + 1) * batch_size]
            feed = {x: batch}
            if corruption: feed[xc] = corrupt(batch, corruption)
            train_step.run(feed_dict=feed)

        if verbose:
            print('Training epoch %d, cost %.2f, r.m.s. loss %.3f ' %
                  (epoch, coder.cost(*valid_args).eval(),
                   coder.rms_loss(validation_set).eval()))
