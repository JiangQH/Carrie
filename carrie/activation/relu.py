from carrie.baselayer import BaseLayer
import numpy as np
class ReLU(BaseLayer):
    """
    the relu layer. y = max(0, x)
    """
    __slots__ = 'mask_'
    def __init__(self):
        pass

    def forward(self, X):
        """
        compute the y = max(x, 0)
        we need to save the mask of x, which is larger than zero
        :param X: input x tensors
        :return: the computed result
        """
        self.__computeMask(X)
        return np.multiply(X, self.mask_)


    def backward(self, Y):
        """
        compute the back diff
        :param Y: backward diff y
        :return:
        """
        return np.multiply(Y, self.mask_)



    def __computeMask(self, X):
        self.mask_ = np.ones_like(X)
        self.mask_[X <= 0] = 0