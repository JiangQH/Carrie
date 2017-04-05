import numpy as np
from carrie.layers.baselayer import BaseLayer
from carrie.utils.im2col import im2col, col2im
from carrie.utils.safty_check import check_gt, check_eq

class BaseConvolution(BaseLayer):
    """
    this is the convolution layer
    """
    def __init__(self, name, kernel_width = 3, kernel_height = 3, kernel_num = 64,
                 pad = 1, stride = 1, w_std=None, b_val=0.1):
        """
        convolution layer params
        :param name: name of this layer
        :param kernel_width: the kernel width
        :param kernel_height: the kernel height
        :param kernel_num: kernel nums
        :param pad: padding to the input
        :param stride: the stride of kernels
        :param w_std: the std to init weight
        :param b_val: the init val of bias
        """
        super(BaseConvolution, self).__init__(name)
        check_gt(kernel_width, 0)
        check_gt(kernel_height, 0)
        check_gt(kernel_num, 0)
        check_gt(pad, 0)
        check_gt(stride, 0)
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_num = kernel_num
        self.pad = pad
        self.stride = stride
        self.w_std = w_std
        self.b_val = b_val
        self.has_init = False


    def initJob(self, bottoms):
        """
        init the weight and bias, note this will be called only once
        during the whole life of program
        :param X:
        :return:
        """
        if self.has_init:
            return
        check_eq(len(bottoms), 1, 'convolution layer should has only one bottom')
        X = bottoms[0]
        # safely check
        check_eq(len(X.shape), 4, 'input shape not agree')
        input_channel = X.shape[1]
        input_height = X.shape[2]
        input_width = X.shape[3]

        # input agree
        check_eq((input_width + 2 * self.pad - self.kernel_width) % self.stride, 0)
        check_eq((input_height + 2 * self.pad - self.kernel_height) % self.stride, 0)

        # save the channel, so we can do safety check later for the weight
        self._input_channel = input_channel

        # the init job for weight and bias
        self.weights = np.random.randn(self.kernel_num, input_channel * self.kernel_height * self.kernel_width)
        self.bias = np.ones((self.kernel_num, 1)) * self.b_val
        self.dw = np.zeros_like(self.weights, dtype=np.float32)
        self.db = np.zeros_like(self.bias, dtype=np.float32)
        if self.w_std is None:
            print 'init convolution layer weights with default...'
            ns = self.kernel_num * input_channel * self.kernel_height * self.kernel_width
            self.weights *= np.sqrt(2.0 / ns)
        else:
            print 'init convolution layer weights with std'.format(self.w_std)
            self.weights *= self.w_std
        self.has_init = True


    def forward(self, bottoms):
        """
        compute the output, using the kernel
        actually we do:
        compute im2col: stretch the input to a matrix, multiply it with weights, and reshape back to y
        :param bottoms: the input ternsors, which should be (n, c, h, w)
        :return: the output convolutioned value, which should be (n, kernel_num, new_h, new_w)
        and new_h = (h + 2*padding - kernel_h) / stride + 1, same with width
        weight should be (k, input_channels, kernel_hegith, kernel_width)
        """
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        # do the forward job, first is the safety check
        check_eq(len(X.shape), 4)
        input_dim = X.shape[0]
        input_channel = X.shape[1]
        input_height = X.shape[2]
        input_width = X.shape[3]
        # input agree
        check_eq((input_width + 2 * self.pad - self.kernel_width) % self.stride, 0)
        check_eq((input_height + 2 * self.pad - self.kernel_height) % self.stride, 0)
        check_eq(input_channel, self._input_channel)
        # now do the forward job
        out_height = (input_height + 2 * self.pad - self.kernel_height) / self.stride + 1
        out_width = (input_width + 2 * self.pad - self.kernel_width) / self.stride + 1
        # get the im2col_data
        col = im2col(X, self.kernel_height, self.kernel_width, self.pad, self.stride)
        out = np.dot(self.weights, col) + self.bias
        out = out.reshape(self.kernel_num, out_height, out_width, input_dim)
        out = out.transpose(3, 0, 1, 2)
        return out



    def backward(self, tops, propagate_down, bottoms):
        """
        abstract the weight and bias update, only do the tiny x job
        note here the bottoms only contains the bottom_shape not the real bottom data
        this is because we not need it at all
        1\ y with respect to x
        need the col to im trick
        :param Y:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(bottoms), 1)
        check_eq(len(propagate_down), 1)
        if propagate_down[0]:
            y = tops[0]
            X_shape = bottoms[0]
            # with respect to x
            dout = y.transpose(1, 2, 3, 0).reshape(self.kernel_num, -1)
            dx_col = self.weights.T * dout
            dx = col2im(dx_col, X_shape, self.kernel_height, self.kernel_width, self.pad, self.stride)
            return dx






