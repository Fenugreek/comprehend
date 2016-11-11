"""
Some utilities for manipulating feature arrays trained from datasets.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import division
import numpy as np
from tsp_solver import greedy as tsp
from tamarind.functions import unit_scale
from tamarind import integers


def corrsort(features, use_tsp=False):
    """
    Given a 2D array, one row per feature, return row indices such that
    adjacent indices correspond to features that are correlated.

    cf. Traveling Salesman Problem. Not an optimal solution.

    use_tsp:
    Use tsp solver. See tsp_solver.greedy module that is used for this.
    Slows run-time considerably: O(N^4) computation, O(N^2) memory.

    Without use_tsp, both computation and memory are O(N^2).
    """

    correlations = np.ma.corrcoef(features)
    if use_tsp: return tsp.solve_tsp(-correlations)

    size = features.shape[0]    
    correlations.mask[np.diag_indices(size)] = True
    
    # initialize results with the pair with the highest correlations.
    largest = np.argmax(correlations)
    results = [int(largest / size), largest % size]
    correlations.mask[:, results[0]] = True

    while len(results) < size:
        correlations.mask[:, results[-1]] = True
        results.append(np.argmax(correlations[results[-1]]))
            
    return results


def tile(X, shape=None, tile=None, spacing=(1, 1), scale=False, bytes=False,
         sort=False, use_tsp=False, spacing_value=0):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    Adapted from deeplearning.net.
    
    X: a 2-D array or a 3-D array.
    a 2-D/3-D array in which every row is a flattened or unflattened image.

    shape: tuple; (height, width)
    The original shape of each image, if X is 2-D; defaults to be most square.

    tile: tuple; (rows, cols)
    The number of images to tile (rows, cols); defaults to be most square.

    bytes:
    If True, return output as uint8 rather than floats.

    scale:
    If True, scale values to [0,1].

    sort:
    If True, tile features such that correlated features appear together
    (ordered columnwise, then rowwise).

    use_tsp:
    Use traveling salesman problem solver for above.
    """

    if shape is None:
        if X.ndim == 3: shape = X.shape[1:]
        else: shape = integers.squarest_factors(X.shape[-1])
    else: assert len(shape) == 2

    if tile is None:
        tile = integers.squarest_factors(X.shape[0], shape)
    else: assert len(tile) == 2
    
    assert len(spacing) == 2

    if sort is True:
        if X.ndim == 2: X = X[corrsort(X, use_tsp=use_tsp)]
        else: X = X[corrsort(X.reshape((len(X), -1)), use_tsp=use_tsp)]
        
    H, W = shape
    Hs, Ws = spacing

    out_shape = [(ishp + tsp) * tshp - tsp
                 for ishp, tshp, tsp in zip(shape, tile, spacing)]
    dt = 'uint8' if bytes else X.dtype
    out_array = np.ones(out_shape, dtype=dt) * spacing_value

    for tile_row in range(tile[0]):
        for tile_col in range(tile[1]):
            if tile_row * tile[1] + tile_col >= X.shape[0]: continue

            this_img = X[tile_row * tile[1] + tile_col]
            if X.ndim == 2: this_img = this_img.reshape(shape)
            if scale: this_img = unit_scale(this_img)

            # add the slice to the corresponding position of  output array
            out_array[tile_row * (H + Hs): tile_row * (H + Hs) + H,
                      tile_col * (W + Ws): tile_col * (W + Ws) + W] = \
                this_img * 255 if bytes else this_img
                
    return out_array
