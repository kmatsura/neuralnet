import numpy as np


def sigmoid(x):
    return 1 / (1. + np.exp(-x))


def sigmoid_diff(x):
    return 1 / (np.exp(x/2) + np.exp(-x/2))**2


def leaky_relu(x, a=0.01):
    if x > 0.:
        return x
    else:
        return a * x


def leaky_relu_diff(x, a=0.01):
    if x > 0:
        return 1
    else:
        return a


def relu(x):
    if x > 0.:
        return x
    else:
        return 0.


def relu_diff(x):
    if x > 0.:
        return 1
    else:
        return 0.
