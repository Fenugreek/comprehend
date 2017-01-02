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
import functions
from tamarind import arrays


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


def get_costs(coder, dataset, batch_size=100, costing=functions.cross_entropy,
              bptt=None):
    """
    Return average cost function value and rms-loss value on validation
    set by coder object with its current weights.
    """

    batch = tf.placeholder(coder.dtype, name='batch',
                           shape=[batch_size] + list(coder.input_shape()[1:]))
    coder.reset_state()
    output = coder.recode(batch, store=bptt)
    if costing != tf.squared_difference:
        inputs = tf.placeholder(coder.dtype, name='inputs',
                                   shape=[batch_size] + list(dataset.shape[1:]))
        predicted = tf.placeholder(coder.dtype, name='predicted',
                                   shape=[batch_size] + list(dataset.shape[1:]))
        cost = costing(inputs, predicted)
        
    n_batches = dataset.shape[0] // batch_size
    if bptt: n_batch_seqs = dataset.shape[2] // bptt
    sum_loss = 0
    sum_cost = 0
    for index in range(n_batches):
        batch_i = dataset[index * batch_size : (index+1) * batch_size]
#        print(index)
        if bptt:
            recoded = []
            for seq in range(n_batch_seqs):
                batch_seq = batch_i[:, :, seq * bptt : (seq+1) * bptt]
                recoded.append(output.eval(feed_dict={batch: batch_seq}))
            coder.reset_state()
            recoded = np.concatenate(recoded, axis=2)
        else: recoded = output.eval(feed_dict={batch: batch_i})
        
        sum_loss += np.mean(np.mean(arrays.plane((batch_i - recoded)**2),
                                    axis=1)**.5)
        if costing != tf.squared_difference:
            costed = cost.eval(feed_dict={inputs: batch_i, predicted: recoded})
            sum_cost += np.mean(costed)
                
    if costing == tf.squared_difference: sum_cost = sum_loss    
    return (sum_cost / n_batches, sum_loss / n_batches)


def get_label_costs(coder, dataset, labels, batch_size=100):
    """
    Return average cross entropy loss and class error rate on
    dataset by coder object with its current weights.
    """
    
    n_batches = dataset.shape[0] // batch_size
    error = 0.
    cost = 0.
    for index in range(n_batches):
        batch = dataset[index * batch_size : (index+1) * batch_size]
        labels_batch = labels[index * batch_size : (index+1) * batch_size]
        predicted = coder.get_hidden_values(batch)
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(predicted,
                                                              labels_batch)
        cost += tf.reduce_mean(loss).eval()

        bad_prediction = tf.not_equal(tf.argmax(predicted , 1), labels_batch)
        error += tf.reduce_mean(tf.cast(bad_prediction, tf.float32)).eval()
    
    return (cost / n_batches, error / n_batches)


def get_target_costs(coder, dataset, targets, batch_size=100, costing=tf.squared_difference):
    """
    Return average r.m.s. error between target and hidden values on 
    dataset by coder object with its current weights.
    """

    batch = tf.placeholder(coder.dtype, [batch_size] + list(dataset.shape[1:]))
    targets_batch = tf.placeholder(coder.dtype, [batch_size] + list(targets.shape[1:]))

    predicted = coder.get_hidden_values(batch)
    cost = tf.reduce_mean(costing(targets_batch, predicted))
    
    n_batches = dataset.shape[0] // batch_size
    sum_cost = 0.
    for index in range(n_batches):
        feed_dict = {batch: dataset[index * batch_size : (index+1) * batch_size],
                     targets_batch: targets[index * batch_size : (index+1) * batch_size]}
        sum_cost += cost.eval(feed_dict=feed_dict)
    
    ave_cost = sum_cost / n_batches
    if costing == tf.squared_difference: ave_cost **= .5
    return ave_cost


def train(sess, coder, dataset, validation_set, verbose=False,
          training_epochs=10, learning_rate=0.001, batch_size=100,
          costing=functions.cross_entropy, bptt=None):
    """
    Train a networks object on given data.

    sess: TensorFlow session.

    coder: networks object that supports cost() and rms_loss() methods.

    dataset: dataset for training.

    validation_set: dataset for monitoring.
    """

    kwargs = {'function': costing}
    if bptt: kwargs['store'] = True
    cost = coder.cost(*coder.train_args, **kwargs)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    sess.run(tf.global_variables_initializer())    

    if verbose:
        print('Initial cost %.4f r.m.s. loss %.4f' %
              get_costs(coder, validation_set, batch_size, costing, bptt=bptt))
    
    n_train_batches = dataset.shape[0] // batch_size
    if bptt: n_batch_seqs = dataset.shape[2] // bptt
    for epoch in range(training_epochs):

        for index in range(n_train_batches):
#            print(index)
            batch = dataset[index * batch_size : (index+1) * batch_size]
            if bptt:
                for seq in range(n_batch_seqs):
                    batch_seq = batch[:, :, seq * bptt : (seq+1) * bptt]
                    train_step.run(feed_dict=coder.train_feed(batch_seq))
                coder.reset_state()
            else:
                train_step.run(feed_dict=coder.train_feed(batch))
                
        if verbose:
            print('Training epoch %d cost %.4f r.m.s. loss %.4f ' %
                  ((epoch,) + get_costs(coder, validation_set, batch_size, costing, bptt=bptt)))


def label_train(sess, coder, dataset, valid_set, labels, valid_labels,
                verbose=False, training_epochs=10, learning_rate=0.001,
                batch_size=100, costing='dummy'):
    """
    Train a networks object on given data for classification.

    sess: TensorFlow session.

    coder: networks object that supports label_cost() method.

    dataset, labels: dataset for training, with associated labels.

    valid_set, valid_labels: dataset for monitoring, with associated labels.
    """

    train_args = coder.train_args + [tf.placeholder(tf.int32, shape=[None])]
    train_step = tf.train.AdamOptimizer(learning_rate)\
                     .minimize(coder.label_cost(*train_args))
    sess.run(tf.global_variables_initializer())    

    if verbose:
        print('Initial cost %5.2f error rate %.3f ' %
              get_label_costs(coder, valid_set, valid_labels, batch_size))

    n_train_batches = dataset.shape[0] // batch_size
    for epoch in range(training_epochs):

        for index in range(n_train_batches):
            train_feed = coder.train_feed(dataset[index * batch_size :
                                                  (index+1) * batch_size])
            train_feed[train_args[-1]] = labels[index * batch_size :
                                                (index+1) * batch_size]
            train_step.run(feed_dict=train_feed)

        if verbose:
            print('Training epoch %d cost %5.2f error rate %.3f ' %
                  ((epoch,) + get_label_costs(coder, valid_set, valid_labels,
                                              batch_size)))


def target_train(sess, coder, dataset, valid_set, targets, valid_targets,
                 verbose=False, training_epochs=10, learning_rate=0.001,
                 batch_size=100, costing=tf.squared_difference):
    """
    Train a networks object on given data for matching hidden values with targets.

    sess: TensorFlow session.

    coder: networks object that supports get_hidden_values() method.

    dataset, targets: dataset for training, with associated targets.

    valid_set, valid_targets: dataset for monitoring, with associated targets.
    """

    train_args = coder.train_args + [tf.placeholder(coder.dtype,
                                                    shape=[None, coder.n_hidden])]
    train_step = tf.train.AdamOptimizer(learning_rate)\
                     .minimize(coder.target_cost(*train_args, function=costing))
    sess.run(tf.global_variables_initializer())    

    if verbose:
        print('Initial cost/r.m.s error %5.3f' %
              get_target_costs(coder, valid_set, valid_targets, batch_size, costing))

    n_train_batches = dataset.shape[0] // batch_size
    for epoch in range(training_epochs):

        for index in range(n_train_batches):
            train_feed = coder.train_feed(dataset[index * batch_size :
                                                  (index+1) * batch_size])
            train_feed[train_args[-1]] = targets[index * batch_size :
                                                (index+1) * batch_size]
            train_step.run(feed_dict=train_feed)

        if verbose:
            print('Training epoch %d cost/r.m.s. error %5.3f' %
                  (epoch, get_target_costs(coder, valid_set, valid_targets,
                                            batch_size, costing)))
