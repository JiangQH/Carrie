from baselayer import BaseLayer
from carrie.math.math import softmax
import numpy as np
from carrie.utils.safty_check import check_eq
class Softmax(BaseLayer):
    """
    this is the softmax layer, compute the softmax output to get a prob for the input
    note here in-order to avoid numerical problems, we need to subtract the x_max
    """

    def __init__(self, name):
        super(Softmax, self).__init__(name)


    def forward(self, bottoms):
        """
        :param bottoms:
        :return:
        """
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        return softmax(X)


    def backward(self, tops, propagate_down, bottoms):
        """
        :param tops:
        :param propagate_down:
        :param bottoms:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(bottoms), 1)
        check_eq(len(propagate_down), 1)
        X = bottoms[0]
        y = tops[0]
        if propagate_down[0]:
            val = softmax(X)
            jac = np.diag(val) - np.dot(val, val.T)
            return np.multiply(np.sum(jac, axis=1), y)






