from carrie.layers.baselayer import BaseLayer
from carrie.utils.safty_check import check_eq, check_gt
from carrie.utils.constant import *
from carrie.utils.im2col import im2col
import numpy as np
class Pooling(BaseLayer):

    def __init__(self, name, pooling_type=POOLING_TYPE.MAX, kernel_width=2,
                 kernel_height=2, stride=2, pad=PADDING_TYPE.VALID):
        super(Pooling, self).__init__(name)
        # safety check
        assert isinstance(pooling_type, POOLING_TYPE)
        assert isinstance(pad, PADDING_TYPE)
        check_gt(kernel_width, 0)
        check_gt(kernel_height, 0)
        check_gt(stride, 0)
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.pooling_type = POOLING_TYPE
        self.pad = pad
        self.stride = stride


    def forward(self, bottoms):
        """
        forward pass
        :param bottoms:
        :return:
        """
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        if self.pooling_type == POOLING_TYPE.MAX:
            def maxpool(X_col):
                max_idx = np.argmax(X_col, axis=0)
                out = X_col[max_idx, range(max_idx.size)]
                self.max_idx = max_idx
                return out
            return self._pool_forward(X, maxpool)
        elif self.pooling_type == POOLING_TYPE.AVG:
            def avgpool(X_col):
                out = np.mean(X_col, axis=0)
                return out
            return self._pool_forward(X, avgpool)


    def backward(self, tops, propagate_down, bottoms):
        """
        backward pass
        :param tops:
        :param propagate_down:
        :param bottoms:
        :return:
        """
        check_eq(len(bottoms), 1)
        check_eq(len(tops), 1)
        check_eq(len(propagate_down), 1)
        if propagate_down[0]:
            dout = tops[0]
            if self.pooling_type == POOLING_TYPE.MAX:
                def dmaxpool(dx_col, dout_col):
                    dx_col[self.max_idx, range(dout_col.size)] = dout_col
                    return dx_col
                return self._pool_backward(dout, dmaxpool)
            elif self.pooling_type == POOLING_TYPE.AVG:
                def davgpool(dx_col, dout_col):
                    dx_col[:, range(dout_col.size)] = 1. / dx_col.shape[0] * dout_col
                    return dx_col
                return self._pool_backward(dout, davgpool)

    def _pool_forward(self, X, pool_fun):
        """
        the actually forward function, call the pool_fun
        :param X:
        :param pool_fun:
        :return:
        """
        check_eq(len(X.shape), 4)
        n, k, h, w = X.shape
        pad = 0
        out_h = (h - self.kernel_height) / self.stride + 1
        out_w = (w - self.kernel_width) / self.stride + 1
        # get the output stride
        if self.pad == PADDING_TYPE.SAME:
            check_eq((h - self.kernel_height) % self.stride, 0, msg='cannot broadcast with same pad, input not agree')
            check_eq((w - self.kernel_width) % self.stride, 0, msg='cannot broadcast with same pad, input not agree')

        elif self.pad == PADDING_TYPE.VALID:
            pad_h = self.stride - (h - self.kernel_height) % self.stride
            pad_w = self.stride - (h - self.kernel_width) % self.stride
            check_eq(pad_h % 2, 0, msg='cannot pad to valid')
            check_eq(pad_w % 2, 0, msg='cannot pad to valid')
            pad_h /= 2
            pad_w /= 2
            pad = pad_h
            out_h = (h - self.kernel_height + 2 * pad_h) / self.stride + 1
            out_w = (w - self.kernel_width + 2 * pad_w) / self.stride + 1

        X_reshaped = X.reshape(n * k, 1, h, w)
        X_col = im2col(X_reshaped, kernel_height=self.kernel_height, kernel_width=self.kernel_width,
                       pad=pad, stride=self.stride)
        out = pool_fun(X_col)
        out = out.reshape(out_h, out_w, n, k)
        out = out.transpose(2, 3, 0, 1)
        return out


    def _pool_backward(self, dout, dpool_fun):
        """
        the actually backward function, call the dpool_fun
        :param dout:
        :param dpool_fun:
        :return:
        """
        