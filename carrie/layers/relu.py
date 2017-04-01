import numpy as np
from carrie.layers.baselayer import BaseLayer
from carrie.utils.safty_check import check_eq

class ReLU(BaseLayer):
    """
    the relu layer. y = max(0, x)
    """
    __slots__ = 'mask_'
    def __init__(self, name):
        super(ReLU, self).__init__(name)


    def forward(self, bottoms):
        """
        :param bottoms:
        :return:
        """
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        self.__computeMask(X)
        return np.multiply(X, self.mask_)


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
        if propagate_down[0]:
            y = tops[0]
            return np.multiply(y, self.mask_)



    def __computeMask(self, X):
        self.mask_ = np.ones_like(X)
        self.mask_[X <= 0] = 0