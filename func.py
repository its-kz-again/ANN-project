import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0, Z)
    return A


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    return dZ * dA


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def test():
    print("test")
