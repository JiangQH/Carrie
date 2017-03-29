from carrie.layers.baselayer import BaseLayer
from carrie.math import math
class Tanh(BaseLayer):
    """
    the tanh layer
    """
    __slots__ = 'input_'
    def __init__(self):
        pass

    def forward(self, X):
        """
        compute 2* sigmoid(2x) - 1
        :param X:
        :return:
        """
        self.input_ = X
        return 2 * math.sigmoid(2 * X) - 1


    def backward(self, Y):
        """
        compute backward. 4*sigmoid(2x)*(1-sigmoid(2x))
        :param Y:
        :return:
        """
        return Y * 4 * math.sigmoid(2 * self.input_) * (1 - math.sigmoid(2 * self.input_))
