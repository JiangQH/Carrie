from carrie.baselayer import BaseLayer
import numpy as np
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
    def __init__(self):
        pass

    def forward(self, X):
        """
        compute 1 / (1 + exp(-x))
        :param X: the input tensor
        :return: forward results
        """
        self.input_ = X
        return  math.sigmoid(X)


    def backward(self, Y):
        """
        compute backward which is s(x)(1-s(x)) * Y
        :param Y: the top tensor, flowed diff
        :return: backward result
        """
        val = math.sigmoid(self.input_) * (1 - math.sigmoid(self.input_))
        return Y * val



