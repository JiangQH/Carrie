from base_convolution import BaseConvolution
import numpy as np
from carrie.utils.im2col import im2col
from carrie.utils.safty_check import check_eq
class Convolution(BaseConvolution):
    def __init__(self, name, kernel_width = 3, kernel_height = 3, kernel_num = 64,
                 pad = 1, stride = 1, w_std=None, b_val=0.1):
        """
        just call the parent
        :param name:
        :param kernel_width:
        :param kernel_height:
        :param kernel_num:
        :param pad:
        :param stride:
        :param w_std:
        :param b_val:
        """
        super(Convolution, self).__init__(name, kernel_width, kernel_height, kernel_num,
                 pad, stride, w_std, b_val)


    def initJob(self, bottoms):
        """
        :param bottoms:
        :return:
        """
        super(Convolution, self).initJob(bottoms)


    def forward(self, bottoms):
        """
        :param bottoms:
        :return:
        """
        return super(Convolution, self).forward(bottoms)


    def backward(self, tops, propagate_down, bottoms):
        """
        call the base just
        :param Y:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(bottoms), 1)
        check_eq(len(propagate_down), 1)
        y = tops[0]
        X = bottoms[0]
        # update the weight and bias here
        self.db = np.sum(y, axis=(0, 2, 3)).reshape(self.db.shape)
        # update the weight
        dout_r = y.transpose(1, 2, 3, 0).reshape(self.kernel_num, -1)
        col = im2col(X, self.kernel_height, self.kernel_width, self.pad, self.stride)
        self.dw = (dout_r * col.T).reshape(self.dw.shape)
        # now back to x
        return super(Convolution, self).backward(tops, propagate_down, bottoms)