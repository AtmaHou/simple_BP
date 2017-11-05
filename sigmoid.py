#!/usr/bin/env python

import numpy as np


def sigmoid(x):

    if not hasattr(x, "__len__"):
        x = np.array([x])
    if isinstance(x, np.ndarray):
        s = 1 / (1.0 + np.exp(-x))
    else:
        raise TypeError
    return s


def sigmoid_grad(s):
    ds = s * (1 - s)
    return ds


def test_sigmoid_basic():
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)


def test_sigmoid():
    x = 2
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        0.8807970779778823
    ])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        0.10499358540350662
    ])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)

if __name__ == "__main__":
    test_sigmoid_basic()
    test_sigmoid()
