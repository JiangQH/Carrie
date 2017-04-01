from base_convolution import BaseConvolution
from carrie.utils.im2col import im2col
import numpy as np
from carrie.utils.safty_check import check_eq

class Deconvolution(BaseConvolution):

    def __init__(self, name, kernel_width = 3, kernel_height = 3, kernel_num = 64,
                 pad = 1, stride = 1, w_std=None, b_val=0.1):
        super(Deconvolution, self).__init__(name, kernel_width, kernel_height, kernel_num,
                                          pad, stride, w_std, b_val)


    def initJob(self, bottoms):
        """
        :param bottoms:
        :return:
        """
        super(Deconvolution, self).initJob(bottoms)


    def forward(self, bottoms):
        """
        forward pass, do it as backward
        :param X:
        :return:
        """
        # compute the shape
        check_eq(len(bottoms), 1)
        x = bottoms[0]
        [n, c, h, w] = x.shape
        out_h = (h + 2 * self.pad - self.kernel_height) / self.stride + 1
        out_w = (w + 2 * self.pad - self.kernel_width) / self.stride + 1
        out_c = self.kernel_num
        x_shape = [n, out_c, out_h, out_w]
        super(Deconvolution, self).backward(bottoms, [True], [x_shape])

    def backward(self, tops, propagate_down, bottoms):
        """
        the back ward pass, do it as forward
        :param y:
        :param X:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(propagate_down), 1)
        check_eq(len(bottoms), 1)
        y = tops[0]
        X = tops[0]
        # get the im2col of y, which corresponds to
        col = im2col(y, self.kernel_height, self.kernel_width, self.pad, self.stride)
        self.db = np.sum(col, axis=(1, 2, 3)).reshape(self.kernel_num, -1)

        x_reshape = X.transpose(1, 2, 3, 0).reshape(self.kernel_num, -1)
        self.dw = (x_reshape * col.T).reshape(self.dw.shape)

        # now the backward
        super(Deconvolution, self).forward(tops)