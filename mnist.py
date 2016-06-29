"""
Some utilities for manipulating the mnist dataset.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import division

import numpy as np
from numpy.random import randint, random
from matplotlib import pyplot

import tensorflow.examples.tutorials.mnist as mnist_data
import train, features


def get_sample(dataset=None, samples=10, start_idx=0):
    """
    Return <samples> of each digit from dataset[:, <start_dx>],
    in order of digits.

    Dataset defaults to the test dataset.
    """
    if dataset is None:
        dataset = mnist_data.input_data.read_data_sets('MNIST_data').test
        
    X, Y = dataset.images, dataset.labels
    
    results = [[] for i in range(10)]
    incomplete = 10
    idx = start_idx
    while incomplete and idx < len(Y):
        x, y = X[idx], Y[idx]
        count = len(results[y])
        if count < samples:
            results[y].append(idx)
            if count == samples - 1: incomplete -= 1
        idx += 1

    return X[np.array(results).flatten()]


def test_coder(coder, sample, corruption=.3, block_corruption=.2):
    """
    Test a neural network on MNIST sample. Returns a list of
    (input, nn output) images.

    coder:
    Trained neural network object, that supports the recode() method.

    sample:
    MNIST sample for testing, e.g. as returned by mnist_sample.

    corruption:
    Perform a test after corrupting sample by this fraction of pixels,
    which are randomly picked and set to 0.

    block_corruption:
    As before, but corruption is a random rectangle of this size.
    """
    
    if type(sample) == list:
        return [(s, coder.recode(s[0]).eval()) for s in sample]
    
    results = [(sample, coder.recode(sample).eval())]
    size = len(sample[0])
    
    if corruption is not None:
        corrupted = train.corrupt(sample, corruption)
        results.append((corrupted, coder.recode(corrupted).eval()))

    if block_corruption is not None:
        corrupted = block_corrupt(sample, block_corruption)
        results.append((corrupted, coder.recode(corrupted).eval()))

    return results
    

def mosaic(results, show=True, transpose=False):
    """
    Convert output of test_coder() into a 2D image suitable for display.
    Returns the associated 2D array.
    
    show:
    If True, plot the image.
    """

    if transpose: results = np.array(results).swapaxes(0, 1)
    
    data_type = results[0][0].dtype
    divider = np.zeros((3, 307), dtype=data_type)
    divider[1, :] = 1.0
    vtiles = []
    for result in results:
        vtiles.append([features.tile(result[0], spacing=(3,3), bytes=False)])
        for r in result[1:]:
            vtiles[-1].append(divider)
            vtiles[-1].append(features.tile(r, spacing=(3,3), bytes=False))

    mosaic = [np.concatenate(vtiles[0])]
    divider = np.zeros((len(mosaic[0]), 3), dtype=data_type)
    divider[:, 1] = 1.0
    for tiles in vtiles[1:]:
        mosaic.append(divider)
        mosaic.append(np.concatenate(tiles))

    mosaic = np.concatenate(mosaic, axis=1)

    if show:
        pyplot.imshow(mosaic)
        pyplot.show()
        
    return mosaic

    
def block_corrupt(dataX, corruption_level=.1):
    """
    Return a copy of dataX MNIST images after corrupting each row with
    a rectangle of size corruption_level.
    """
    
    count = len(dataX)
    size = dataX[0].size
    length = int(np.sqrt(size))
    corrupt_area = corruption_level * size

    breadths = randint(1, int(np.sqrt(corrupt_area)), count)
    lengths = (corrupt_area / breadths).astype(int)
    switch = randint(0, 2, count)
    breadths[switch==0] = lengths[switch==0]
    lengths = (corrupt_area / breadths).astype(int)

    loc_x = randint(0, length, count)
    loc_y = randint(0, length, count)

    corruptX = np.zeros(dataX.shape, dtype=dataX.dtype)
    for i, img in enumerate(dataX):
        bi, li = breadths[i], lengths[i]
        ind_x = np.arange(loc_x[i], loc_x[i] + bi, dtype=int) % length
        ind_y = np.arange(loc_y[i], loc_y[i] + li, dtype=int) % length
        corrupted = img.copy().reshape((length, length))
        corrupted[(np.tile(ind_x, li),
                   np.repeat(ind_y, bi))] = random(bi * li)
        corruptX[i] = corrupted.reshape(img.shape)

    return corruptX
