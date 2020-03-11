#!/usr/bin/env python

import numpy as np


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    # YOUR CODE HERE
    length = np.sqrt(np.sum(x*x, axis=1, keepdims=True))
    x = np.divide(x, length)
    # END YOUR CODE
    return x


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    # YOUR CODE HERE
    if len(x.shape) > 1:
        # When x is a N-by-D matrix
        max_for_row = np.max(x, axis=1, keepdims=True)
        x = np.subtract(x, max_for_row)
        x = np.exp(x)
        sum_for_row = np.sum(x, axis=1, keepdims=True)
        x = np.divide(x, sum_for_row)

    else:
        # When x is a 1-by-D vector
        x -= np.max(x)
        x = np.exp(x)
        x /= np.sum(x)
    # END YOUR CODE
    return x
