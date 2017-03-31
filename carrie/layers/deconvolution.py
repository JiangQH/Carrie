from base_convolution import BaseConvolution
from carrie.utils.im2col import im2col
import numpy as np

class Deconvolution(BaseConvolution):

    def __init__(self, name, kernel_width = 3, kernel_height = 3, kernel_num = 64,
                 pad = 1, stride = 1, w_std=None, b_val=0.1):
        super(Deconvolution, self).__init__(name, kernel_width, kernel_height, kernel_num,
                                          pad, stride, w_std, b_val)


    def initJob(self, X):
        """
        call the parent pass
        :param X:
        :return:
        """
        super(Deconvolution, self).initJob(X)


    def forward(self, X, y):
        """
        forward pass, do it as backward
        :param X:
        :return:
        """
        super(Deconvolution, self).backward(X, y)

    def backward(self, y, X):
        """
        the back ward pass, do it as forward
        :param y:
        :param X:
        :return:
        """
        # get the im2col of y, which corresponds to
        col = im2col(y, self.kernel_height, self.kernel_width, self.pad, self.stride)
        self.db = np.sum(col, axis=(1, 2, 3)).reshape(self.kernel_num, -1)

        x_reshape = X.transpose(1, 2, 3, 0).reshape(self.kernel_num, -1)
        self.dw = (x_reshape * col.T).reshape(self.dw.shape)

        # now the backward
        super(Deconvolution, self).forward(y, X)