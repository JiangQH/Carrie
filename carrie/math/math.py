import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))


def softmax(x):
    ex = np.exp(x - np.max(x, axis=1))
    return ex / np.sum(ex, axis=1)