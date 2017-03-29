import numpy as np

from carrie.layers.baselayer import BaseLayer


class Convolution(BaseLayer):
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
        assert kernel_width > 0 and kernel_height > 0
        assert kernel_num > 0 and pad > 1 and stride > 1
        self.name = name
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_num = kernel_num
        self.pad = pad
        self.stride = stride
        self.w_std = w_std
        self.b_val = b_val
        self.has_init = False


    def initJob(self, X):
        """
        init the weight and bias, note this will be called only once
        during the whole life of program
        :param X:
        :return:
        """
        if self.has_init:
            return
        # safely check
        assert len(X.shape) == 4, 'input shape not agree'
        input_channel = X.shape[1]
        input_height = X.shape[2]
        input_width = X.shape[3]

        # input agree
        assert (input_width + 2 * self.pad - self.kernel_width) % self.stride == 0, 'input and the pad, ' \
                                                                                    'stride not agree'
        assert (input_height + 2 * self.pad - self.kernel_height) % self.stride == 0, 'input and the pad, ' \
                                                                                      'stride not agree'

        # save the channel, so we can do safety check later for the weight
        self._input_channel = input_channel

        # the init job for weight and bias
        self._weights = np.random.randn((self.kernel_num, input_channel * self.kernel_height * self.kernel_width))
        self._bias = np.ones((self.kernel_num, 1)) * self.b_val
        if self.w_std is None:
            print 'init convolution layer weights with default...'
            ns = self.kernel_num * input_channel * self.kernel_height * self.kernel_width
            self._weights *= np.sqrt(2.0 / ns)
        else:
            print 'init convolution layer weights with std'.format(self.w_std)
            self._weights *= self.w_std
        self.has_init = True


    def forward(self, X):
        """
        compute the output, using the kernel
        actually we do:
        compute im2col: stretch the input to a matrix, multiply it with weights, and reshape back to y
        :param X: the input ternsors, which should be (n, c, h, w)
        :return: the output convolutioned value, which should be (n, kernel_num, new_h, new_w)
        and new_h = (h + 2*padding - kernel_h) / stride + 1, same with width
        weight should be (k, input_channels, kernel_hegith, kernel_width)
        """
        # do the forward job, first is the safety check
        assert len(X.shape) == 4, 'input dim not agree'
        input_dim = X.shape[0]
        input_channel = X.shape[1]
        input_height = X.shape[2]
        input_width = X.shape[3]
        # input agree
        assert (input_width + 2 * self.pad - self.kernel_width) % self.stride == 0, 'input and the pad, ' \
                                                                                    'stride not agree'
        assert (input_height + 2 * self.pad - self.kernel_height) % self.stride == 0, 'input and the pad, ' \
                                                                                      'stride not agree'
        # channel must stay stable, not change
        assert input_channel == self._input_channel, 'input channel and weights not equal, {} vs {}'.format(
            input_channel, self._input_channel
        )

        # now do the forward job
        out_height = (input_height + 2 * self.pad - self.kernel_height) / self.stride + 1
        out_width = (input_width + 2 * self.pad - self.kernel_width) / self.stride + 1
        y = np.zeros((input_dim, self.kernel_num, out_height, out_width))
        for it in range(input_dim):
            data = X[it, ...]
            im2col = self.__im2col(data)
            y[it, ...] = (self._weights * im2col + self._bias).reshape((self.kernel_num, out_height,
                                                                        out_width))
        return y





    def backward(self, Y):
        """
        do the backward job, it compute two things.
        1\ gradient with respect to x
        2\ gradient with respect to weight and bias
        :param Y:
        :return:
        """
        pass


    def __im2col(self, X):
        """
        convert the X to matrix
        im2col function.
        1\ im2col: (input_channel * kernel_height * kernel_width, out_h * out_w)
        :param X: (input_channel, input_height, input_width)
        :return:
        """
        [input_channel, input_height, input_width] = X.shape
        out_height = (input_height + 2 * self.pad - self.kernel_height) / self.stride + 1
        out_width = (input_width + 2 * self.pad - self.kernel_width) / self.stride + 1
        im2col = np.zeros((input_channel * self.kernel_height * self.kernel_width,
                           out_height * out_width))
        # now do the scan-folding job
        h_start = -self.pad
        for h in range(out_height):
            w_start = -self.pad
            for w in range(out_width):
                local_data = np.zeros((input_channel, self.kernel_height, self.kernel_width))
                h_starter = h_start if h_start > 0 else 0
                w_starter = w_start if w_start > 0 else 0
                h_ender = h_start + self.kernel_height \
                    if (h_start + self.kernel_height) < input_height else input_height - 1
                w_ender = w_start + self.kernel_width \
                    if (w_start + self.kernel_width) < input_width else input_width - 1
                local_data[:, h_starter-h_start:h_ender-h_start, w_starter-w_start:w_ender-w_start] \
                    = X[:, h_start:h_ender, w_starter:w_ender]
                # reshape it to a column and assigned to the im2col data
                im2col[:, w + h * out_height] = local_data.flatten().transpose()

                # update w_start
                w_start += self.pad
            # update h_start
            h_start += self.pad
        # return the im2col data
        return im2col


