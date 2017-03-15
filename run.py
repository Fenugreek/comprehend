#!/usr/bin/python
"""
Load/construct various NN models from the commandline, train/run them on data, save to disk.
"""
import os.path, importlib, sys
import argparse, cPickle

import numpy as np
from matplotlib import pyplot
import bloscpack as bp

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as mnist_data
from comprehend import train, features, layers, functions, mnist

from tamarind import logging
import tamarind.functions

# Specify some functions here that are hard to do on the command line.
# ====================================================================

# Cost function for training
cost_fn = functions.cross_entropy

# Activation function for encoding and decoding, respectively.
coding_fn = tf.sigmoid
decoding_fn = coding_fn

# Call this function on NN object after construction
aux_fn = None

# Call these functions on data after loading from disk.
data_fn = None 
target_fn = data_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', metavar='<model>', 
                        help='network architecture to load from networks module. e.g. Auto, Denoising, RBM, RNN, VAE.')
    parser.add_argument('--module', metavar='<module>', default='comprehend.networks',
                        help='find <model> in this module (defaults to module supplied with this library).')
    parser.add_argument('--fromfile', metavar='<filename.dat>',
                        help='previous params.dat file to load from and resume training.')
    parser.add_argument('--add', metavar='<model>',
                        help='network layer to add to architecture loaded fromfile.')

    parser.add_argument('--visible', metavar='N', type=int,
                        help='number of visible units; inferred from data if not specified')
    parser.add_argument('--hidden', metavar='N', type=int, default=500,
                        help='number of hidden units')
    parser.add_argument('--options', metavar='<option>=<integer>[,<integer>...]', nargs='+',
                        help='options to pass to constructor of network architecture; only integers supported for now.')

    parser.add_argument('--data', metavar='<filename.blp>', nargs='+',
                        help='data file(s) to use for training. If multiple, they are joined.')
    parser.add_argument('--labels', metavar='<filename.blp>',
                        help='data file with labels for classification training.')
    parser.add_argument('--targets', metavar='<filename.blp>',
                        help='data file with targets for mapping training.')
    parser.add_argument('--permute', metavar='<filename.blp>',
                        help='Permute order of rows in input data according to indices in this file.')
    parser.add_argument('--prep_data', metavar='<filename.blp>',
                        help="Supply optional auxiliary file here for post-processing data by NN object's prep_data method, if any.")
    parser.add_argument('--validation', metavar='R', type=float, default=.08,
                        help='fraction of dataset to use as validation set.')

    
    parser.add_argument('--batch', metavar='N', type=int, default=300,
                        help='size of each mini-batch')
    parser.add_argument('--bptt', metavar='N', type=int,
                        help='backpropagation through time; no. of timesteps')
    parser.add_argument('--epochs', metavar='N', type=int, default=0,
                        help='No. of epochs to train.')
    parser.add_argument('--learning', metavar='R', type=float, default=0.001,
                        help='learning rate for gradient descent algorithm')
    parser.add_argument('--random_seed', metavar='N', type=int, default=123,
                        help='Seed random number generator with this, for repeatable results.')

    parser.add_argument('--output', metavar='<path/prefix>',
                        help='output params and figures to <path/prefix>{params,features,mosaic}.dat.')
    parser.add_argument('--loss', action='store_true',
                        help='Print r.m.s. loss on validation data.')
    parser.add_argument('--dump', metavar='{hidden|recode|<custom method>}',
                        help='dump computed hidden or recoded values to disk. Needs --output to be specified also.')
    parser.add_argument('--features', action='store_true',
                        help='Print image visualization of weights to disk.')

    parser.add_argument('--log', metavar='log_level', default='info',
                        help='debug, info, warning, error or critical.')
    args = parser.parse_args()
    
    logfile = open(args.output+'log.txt', 'w') if args.output else None
    logger = logging.Logger(__file__.split('/')[-1], args.log,
                            logfile=logfile, printer=sys.stdout)
    verbose = logger.level_value < 30 # INFO or more verbose
    
    if not args.model:
        args.model = 'Layers'
        args.module = 'comprehend.layers'
    if '.' not in args.module: args.module = 'comprehend.' + args.module
    module = importlib.import_module(args.module)
    coder_class = getattr(module, args.model)


    # load data
    # ---------

    # Primary dataset
    if args.data is None:
        #load mnist as default
        train_data = mnist_data.read_data_sets('MNIST_data').train
        datas = [train_data.images]
        if hasattr(coder_class, 'prep_mnist'):
            datas = [coder_class.prep_mnist(datas[0])]
    else:
        datas = [bp.unpack_ndarray_file(fname) for fname in args.data]
    if data_fn: datas = data_fn(datas)
    train_mode = 'recode'
    
    # auxiliary optional datasets
    if args.labels:
        labels = bp.unpack_ndarray_file(args.labels)
        datas.append(labels)
        train_mode = 'label'
    if args.targets:
        targets = bp.unpack_ndarray_file(args.targets)
        if target_fn: targets = target_fn(targets)
        datas.append(targets)
        train_mode = 'target'
    

    # instantiate neural network object
    # ---------------------------------

    kwargs = {}
    for option in args.options or []:
        key, value = option.split('=')
        if ',' in value: kwargs[key] = [int(v) for v in value.split(',')]
        else: kwargs[key] = int(value)
    if args.bptt: kwargs['seq_length'] = args.bptt

    np.random.seed(args.random_seed)
    coder_class = getattr(module, args.model)
    if args.fromfile:
        coder = coder_class(fromfile=args.fromfile, batch_size=args.batch, verbose=verbose,
                            coding=coding_fn, decoding=decoding_fn, **kwargs)
    else:
        coder = coder_class(n_hidden=args.hidden, n_visible=args.visible or datas[0].shape[1],
                            verbose=verbose, batch_size=args.batch,
                            coding=coding_fn, decoding=decoding_fn, **kwargs)
    if args.add:
        if '.' not in args.add: args.add = 'networks.' + args.add
        module = args.add[:args.add.rfind('.')]
        if '.' not in module: module = 'comprehend.' + module
        module = importlib.import_module(module)
        coder_class = getattr(module, args.add.split('.')[-1], None)
        coder2 = coder_class(n_hidden=args.hidden, n_visible=args.visible or datas[0].shape[1],
                             verbose=verbose, batch_size=args.batch,
                             coding=coding_fn, decoding=decoding_fn, **kwargs)
        coder = layers.add_layer(coder, coder2)
    if aux_fn: aux_fn(coder)

    if args.labels or args.targets:
        n_out = targets.shape[1] if args.targets else np.max(labels) + 1
        if coder.n_hidden != n_out:
            logger.warning("Adding {} output layer to match target", n_out)
            coder = layers.add_class_layer(coder, n_out)


    # Final data prep, possibly dependent on cooder object so here
    # ------------------------------------------------------------
    
    if hasattr(coder, 'prep_data'):
        logger.info("Prepping data...")
        datas = coder.prep_data(datas, args.prep_data)
    
    if args.permute:
        if os.path.isfile(args.permute):
            perm = bp.unpack_ndarray_file(args.permute)
        else:
            logger.warning("Given permutation file does not exits; creating " +
                           args.permute)
            perm = np.random.permutation(len(datas[0]))
            bp.pack_ndarray_file(perm, args.permute)
        for d in datas:
            d[:] = d[perm]
        
    # partition to training and validation
    train_idx = int((1.0 - args.validation) * len(datas[0]))
    logger.info('Train samples {}, validation samples {}',
                train_idx, len(datas[0]) - train_idx)
        
    # now execute requested actions
    # -----------------------------

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())    
    with sess.as_default():
        if args.epochs:
            train.train(sess, coder, datas, train_idx=train_idx, epochs=args.epochs,
                        batch_size=args.batch, learning_rate=args.learning,
                        bptt=args.bptt, logger=logger, cost_fn=cost_fn, mode=train_mode)
        
        if args.output is not None:
            if args.epochs or not args.fromfile:
                coder.save_params(args.output+'params.dat')
                f = open(args.output+'args.txt', 'w')
                cPickle.dump(vars(args), f)
                f.close()
            if args.dump:
                coder.dump_output(datas[0], args.output+args.dump+'.dat',
                                  kind=args.dump, batch_size=args.batch,
                                  overlay=[0, 1])

        if args.loss:
            rms_fn = lambda x, y: tf.reduce_mean(tf.squared_difference(x, y),
                                                 axis=range(1, coder.input_dims))**.5
            train_args = coder.train_args if hasattr(coder, 'train_args') else \
                         coder.init_train_args(mode=train_mode)
            cost = coder.mode_cost(train_mode, *train_args, function=cost_fn)
            loss = coder.mode_cost(train_mode, *train_args, function=rms_fn)

            for count, sli in enumerate([slice(train_idx, None),
                                         slice(0, len(datas[0])-train_idx)]):
                subset = [d[sli] for d in datas]
                print "Cost, loss on %s samples: %.4f %.4f" % \
                      ('subset train' if count else 'validation',
                       train.get_mean_cost(cost, train_args, subset, batch_size=args.batch),
                       train.get_mean_cost(loss, train_args, subset, batch_size=args.batch))
                   

        if args.features and hasattr(coder, 'features'):
            results = coder.features(datas[0][train_idx:]).squeeze()
            results = results[features.corrsort(results, use_tsp=True)]
            if args.data is None: #mnist. Prepare to tile weights.
                results = results.reshape((-1, 28, 28))
            if results.ndim == 3:
                results = features.tile(results, scale=True)
            if args.output is None:
                pyplot.imshow(results, interpolation='nearest')
                pyplot.show()
            else:
                pyplot.imsave(args.output+'features.png', results, origin='upper')

            if hasattr(coder, 'stimuli'):
                results = coder.all_stimuli(-1)
                if args.output is None:
                    pyplot.imshow(results, interpolation='nearest')
                    pyplot.show()
                else:
                    pyplot.imsave(args.output+'stimuli.png', results, origin='upper')

            if args.data is None: #mnist. Perform additional visualization.
                results = mnist.test_coder(coder, datas[0][train_idx:train_idx+100])
                img = mnist.mosaic(results, show=args.output is None)
                if args.output:
                    pyplot.imsave(args.output+'mosaic.png', img, origin='upper')


    sess.close()
    if logfile is not None: logfile.close()
