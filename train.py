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

    shape = [batch_size] + list(coder.input_shape()[1:])
    batch = tf.placeholder(coder.dtype, name='batch', shape=shape)

    coder.reset_state()
    output = coder.recode(batch, store=bptt)

    if costing != tf.squared_difference:
        inputs = tf.placeholder(coder.dtype, name='inputs', shape=shape)
        predicted = tf.placeholder(coder.dtype, name='predicted', shape=shape)
        cost = costing(inputs, predicted)
        
    n_batches = dataset.shape[0] // batch_size
    if bptt: n_batch_seqs = dataset.shape[2] // bptt

    sum_loss = 0
    sum_cost = 0
    for index in range(n_batches):
        batch_i = dataset[index * batch_size : (index+1) * batch_size]

        if bptt:
            recoded = []
            for seq in range(n_batch_seqs):
                batch_seq = batch_i[:, :, seq * bptt : (seq+1) * bptt]
                recoded.append(output.eval(feed_dict={batch: batch_seq}))
                if costing != tf.squared_difference:
                    costed = cost.eval(feed_dict={inputs: batch_seq,
                                                  predicted: recoded[-1]})
                    sum_cost += np.mean(costed) / n_batch_seqs

            coder.reset_state()
            recoded = np.concatenate(recoded, axis=2)
        else: recoded = output.eval(feed_dict={batch: batch_i})
        
        sum_loss += np.mean(np.mean(arrays.plane((batch_i - recoded)**2),
                                    axis=1)**.5)
        if costing != tf.squared_difference and not bptt:
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
    predicted = coder.get_hidden_values(batch)
    if costing != tf.squared_difference:
        shape = [batch_size] + list(targets.shape[1:])
        batch_h = tf.placeholder(coder.dtype, name='batch_h', shape=shape)
        batch_t = tf.placeholder(coder.dtype, name='batch_t', shape=shape)
        cost = tf.reduce_mean(costing(batch_h, batch_t))
    
    n_batches = dataset.shape[0] // batch_size
    sum_cost = 0.
    sum_loss = 0
    for index in range(n_batches):
        hidden = predicted.eval(feed_dict={batch: dataset[index * batch_size :
                                                          (index+1) * batch_size]})
        actual = targets[index * batch_size : (index+1) * batch_size]
        sum_loss += np.mean(np.mean(arrays.plane((actual - hidden)**2),
                                    axis=1)**.5)
        
        if costing != tf.squared_difference:
            costed = cost.eval(feed_dict={batch_h: hidden, batch_t: actual})
            sum_cost += np.mean(costed)

    if costing == tf.squared_difference: sum_cost = sum_loss    
    return (sum_cost / n_batches, sum_loss / n_batches)


def get_trainer(cost, learning_rate=.001, grad_clips=(-1, 1)):
    """Return opertation that trains parameters, given cost tensor."""

    opt = tf.train.AdamOptimizer(learning_rate)
    if grad_clips is None: return opt.minimize(cost)

    grads_vars = []
    for grad_var in opt.compute_gradients(cost):
        if grad_var[0] is None:
#            print 'Warning: no gradient for variable %s' % grad_var[1].name
            continue
        grads_vars.append((tf.clip_by_value(grad_var[0], -1., 1.), grad_var[1]))

    return opt.apply_gradients(grads_vars)


def train(sess, coder, dataset, validation_set, verbose=False,
          training_epochs=10, learning_rate=0.001, batch_size=100,
          costing=functions.cross_entropy, bptt=None, skips=None):
    """
    Train a networks object on given data.

    sess: TensorFlow session.

    coder: networks object that supports cost() and rms_loss() methods.

    dataset: dataset for training.

    validation_set: dataset for monitoring.
    """

    kwargs = {'function': costing, 'skips': skips}
    if bptt: kwargs['store'] = True
    cost = coder.cost(*coder.train_args, **kwargs)
    opt = tf.train.AdamOptimizer(learning_rate)
    
    train_step = get_trainer(cost, learning_rate=learning_rate)
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
    train_step = get_trainer(coder.label_cost(*train_args),
                             learning_rate=learning_rate)
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
    train_step = get_trainer(coder.target_cost(*train_args, function=costing),
                             learning_rate=learning_rate)
    sess.run(tf.global_variables_initializer())    

    if verbose:
        print('Initial cost %.4f r.m.s. loss %.4f' %
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
            print('Training epoch %d cost %.4f r.m.s. loss %.4f ' %
                  ((epoch,) + get_target_costs(coder, valid_set, valid_targets,
                                               batch_size, costing)))


def rnn_extend(sess, coder, inputs, skips=None, length=1):
    """
    inputs is batch_size x n_consecutive_seqs x n_visible x seq_length.
    """
    
    shape = inputs.shape
    n_seqs =  shape[1]
    batch = tf.placeholder(coder.dtype, name='batch_seq',
                           shape=(shape[0], shape[2], shape[3]))

    coder.reset_state()
    output = coder.recode(batch, store=True, skips=skips)

    outputs = []
    for index in range(n_seqs):
        batch_seq = inputs[:, index, :, :]
        o, s = sess.run([output, coder.get_state()], feed_dict={batch: batch_seq})
        outputs.append(o)


    outputs.append(coder.predict_sequence(None,
                                          s['hidden'],
                                          length=length*shape[3]).eval())

    return np.array(outputs)
