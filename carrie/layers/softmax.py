from baselayer import BaseLayer
from carrie.math.math import softmax
import numpy as np
class Softmax(BaseLayer):
    """
    this is the softmax layer, compute the softmax output to get a prob for the input
    note here in-order to avoid numerical problems, we need to subtract the x_max
    """

    def __init__(self, name):
        super(Softmax, self).__init__(name)


    def forward(self, X, y):
        """
        note that x_i must be a scaler here, which is the channel dimension
        :param X:
        :param y:
        :return:
        """
        return softmax(X)


    def backward(self, y, X):
        """
        :param y: the top_diff
        :param X: the x data
        :return:
        """
        val = softmax(X)
        jac = np.diag(val) - np.dot(val, val.T)
        return np.multiply(np.sum(jac, axis=1), y)






