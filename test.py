#!/usr/bin/python
"""
Train various TensorFlow models from the commandline.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
import argparse, cPickle

import numpy as np
from matplotlib import pyplot

import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist
import networks, train, features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', metavar='<model>',
                       help='TensorFlow code to run. e.g. auto')
    parser.add_argument('--params', metavar='<filename.dat>',
                       help='previous params_final.dat file to load from and resume training.')
    parser.add_argument('--data', metavar='<filename.dat>',
                       help='data file to use for training. Default: MNIST')
    parser.add_argument('--output', metavar='<path/prefix>',
                       help='output params and figure to <path/prefix>params_init.dat, etc.')
    parser.add_argument('--learning_rate', metavar='R', type=float, default=0.01,
                       help='learning rate for gradient descent algorithm')
    parser.add_argument('--batch', metavar='N', type=int, default=100,
                       help='size of each mini-batch')
    parser.add_argument('--hidden', metavar='N', type=int, default=500,
                       help='number of hidden units')
    parser.add_argument('--epochs', metavar='N', type=int, default=10,
                       help='No. of epochs to train.')
    parser.add_argument('--random_seed', metavar='N', type=int, default=123,
                       help='Seed random number generator with this, for repeatable results.')
    parser.add_argument('--verbose', action='store_true', help='print progress')
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    dataset = cPickle.load(open(args.data)) if args.data else \
              mnist.input_data.read_data_sets('MNIST_data').train.images
    
    global coder
    coder = getattr(networks, args.model)\
               (n_hidden=args.hidden, verbose=args.verbose, fromfile=args.params)
    
    sess = tf.Session()
    with sess.as_default():
        train.train(sess, coder, dataset[:50000], dataset[50000:51000],
                    training_epochs=args.epochs, batch_size=args.batch,
                    learning_rate=args.learning_rate,
                    verbose=args.verbose)

        if args.output is not None:
            coder.save_params(args.output+'params_final.dat')

        if hasattr(coder, 'features'):
            results = coder.features(dataset[50000:51000])
            if type(results) is not tuple: results = (results,)
            tiles = features.tile(*results, scale=True)
            if args.output is None:
                pyplot.imshow(tiles)
                pyplot.show()
            else: pyplot.imsave(args.output+'weights.png', tiles, origin='upper')
    
    sess.close()
    
