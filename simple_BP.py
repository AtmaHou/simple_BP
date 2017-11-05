# coding: utf-8
import numpy as np
import random

from Dataset import Dataset
from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive


Gaussian_setting_config = [
    {
        'mean': [0, 0, 0],
        'cov': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'size': 1000
    },
    {
        'mean': [0, 1, 0],
        'cov': [[1, 0, 1], [0, 2, 2], [1, 2, 5]],
        'size': 1000
    },
    {
        'mean': [-1, 0, 1],
        'cov': [[2, 0, 0], [0, 6, 0], [0, 0, 1]],
        'size': 1000
    },
    {
        'mean': [0, 0.5, 1],
        'cov': [[2, 0, 0], [0, 1, 0], [0, 0, 3]],
        'size': 1000
    },
]

nn_shape = [3, 3, 4]


def forward_backward_prop(data, labels, params, dimensions):
    ofs = 0
    Dx, H, Dy = dimensions

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    z1 = np.dot(data, W1) + b1  # according to broadcast rule, b1 will extend to M dimensions
    h = sigmoid(z1)
    z2 = np.dot(h, W2) + b2
    y_hat = softmax(z2)

    cost = - np.sum(np.log(y_hat[labels == 1])) / data.shape[0]
    d3 = (y_hat - labels) / data.shape[0]
    gradW2 = np.dot(h.T, d3)
    gradb2 = np.sum(d3, 0, keepdims=True)
    dh = np.dot(d3, W2.T)
    grad_h = sigmoid_grad(h) * dh
    gradW1 = np.dot(data.T, grad_h)
    gradb1 = np.sum(grad_h, 0)

    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return y_hat, cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)