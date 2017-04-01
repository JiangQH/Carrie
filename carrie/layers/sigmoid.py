from carrie.layers.baselayer import BaseLayer
from carrie.math import math
from carrie.utils.safty_check import check_eq
class Sigmoid(BaseLayer):
    """
    the Sigmoid function layer
    compute the 1 / (1 + exp(-x))
    since we need the input to compute
    diff, so we must hold the input value
    not change, do not do it inplace
    """

    __slots__ = 'input_'
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, bottoms):
        """
        compute 1 / (1 + exp(-x))
        :param X: the input tensor
        :return: forward results
        """
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        return math.sigmoid(X)


    def backward(self, tops, propagate_down, bottoms):
        """
        :param tops:
        :param propagate_down:
        :param bottoms:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(propagate_down), 1)
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        y = tops[0]
        if propagate_down[0]:
            val = math.sigmoid(X) * (1 - math.sigmoid(X))
            return y * val



