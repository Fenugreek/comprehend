#!/usr/bin/python
"""
Train various TensorFlow models from the commandline, (optionally) saving params learnt to disk.
"""
import argparse, cPickle

import numpy as np
from matplotlib import pyplot

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as mnist_data
import networks, train, features, mnist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', metavar='<model>',
                       help='network architecture to load from networks module. Auto, Denoising, RBM or RNN.')
    parser.add_argument('--params', metavar='<filename.dat>',
                       help='previous params_final.dat file to load from and resume training.')
    parser.add_argument('--data', metavar='<filename.dat>',
                       help='data file to use for training. Default: MNIST')
    parser.add_argument('--output', metavar='<path/prefix>',
                       help='output params and figures to <path/prefix>{params,features,mosaic}.dat.')
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
    parser.add_argument('--use_tsp', action='store_true',
                        help='Use Traveling Salesman Problem solver when arranging features for display (takes time).')
    parser.add_argument('--mosaic', action='store_true', help="Display learnt model's reconstruction of corrupted input.")
    parser.add_argument('--verbose', action='store_true', help='print progress')
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    train_data = cPickle.load(open(args.data)) if args.data else \
                 mnist_data.read_data_sets('MNIST_data').train
    dataset = train_data.images
    
    global coder
    coder = getattr(networks, args.model)\
               (n_hidden=args.hidden, verbose=args.verbose, fromfile=args.params)
    if hasattr(coder, 'prep_mnist'):
        # massage data into shape liked by coder object.
        dataset = coder.prep_mnist(dataset)
    
    sess = tf.Session()
    with sess.as_default():
        train.train(sess, coder, dataset[:50000], dataset[50000:51000],
                    training_epochs=args.epochs, batch_size=args.batch,
                    learning_rate=args.learning_rate,
                    verbose=args.verbose)

        if args.output is not None:
            coder.save_params(args.output+'params.dat')

        if hasattr(coder, 'features'):
            results = coder.features(dataset[50000:51000])
            if type(results) is not tuple: results = (results,)
            tiles = features.tile(*results, scale=True, corr=True,
                                  use_tsp=args.use_tsp)
            if args.output is None:
                pyplot.imshow(tiles)
                pyplot.show()
            else:
                pyplot.imsave(args.output+'features.png', tiles, origin='upper')

        if args.mosaic:
            sample = mnist.get_sample(train_data, start_idx=50000)
            if hasattr(coder, 'prep_mnist'): sample = coder.prep_mnist(sample)
            results = mnist.test_coder(coder, sample)
            img = mnist.mosaic(results, show=args.output is None)
            if args.output:
                pyplot.imsave(args.output+'mosaic.png', img, origin='upper')

    sess.close()
    
