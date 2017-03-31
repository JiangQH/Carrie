from carrie.layers.baselayer import BaseLayer
from carrie.math import math
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

    def forward(self, X, y):
        """
        compute 1 / (1 + exp(-x))
        :param X: the input tensor
        :return: forward results
        """
        return  math.sigmoid(X)


    def backward(self, y, X):
        """
        compute backward which is s(x)(1-s(x)) * Y
        :param Y: the top tensor, flowed diff
        :return: backward result
        """
        val = math.sigmoid(X) * (1 - math.sigmoid(X))
        return y * val



