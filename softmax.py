import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x = x - np.min(x, axis=1)[:, np.newaxis]
        exp_matrix = np.exp(x)
        sum_lst = np.sum(exp_matrix, axis=1, dtype='float64')
        x = exp_matrix / sum_lst[:, np.newaxis]
    else:
        # Vector
        x = x - np.min(x)
        exp_matrix = np.exp(x)
        sum_value = np.sum(exp_matrix, dtype='float64')
        x = exp_matrix / sum_value

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)



def test_softmax():
    print "Running your tests..."
    test4 = softmax(np.array([[2]]))
    print test4
    ans4 = np.array([1])
    assert np.allclose(test4, ans4, rtol=1e-05, atol=1e-06)


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
