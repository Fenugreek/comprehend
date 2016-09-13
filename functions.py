"""
Utility math/stats functions.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import tensorflow as tf

def cross_entropy(x, y, eps=1e-8):
    return -x * tf.log(y + eps) - (1 - x) * tf.log(1 - y + eps)


def logit(data, eps=1e-8):
    """Inverse of the sigmoid function."""

    return -tf.log(1 / (data + eps) - 1 + eps)
