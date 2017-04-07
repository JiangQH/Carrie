import numpy as np
from carrie.layers.baselayer import BaseLayer
from carrie.utils.safty_check import check_eq, check_gt

class Fc(BaseLayer):

    """
    this is the fully connected layer
    """
    def __init__(self, name, num_output, w_std=None, b_val=0.1):
        """
        fully connected layer params
        :param name:
        :param num_output:
        """
        super(Fc, self).__init__(name)
        check_gt(num_output, 1, msg='num output should larger than 1')
        self.num_output = num_output
        self.w_std = w_std
        self.b_val = b_val
        self.has_init = False


    def initJob(self, bottoms):
        """
        the init job
        :param bottoms: the list of bottom iterms, should be one. [m, n]
        :return:
        """
        if self.has_init:
            return
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        # safety check
        assert len(X) == 4 or len(X) == 2
        if len(X) == 4:
            X = np.reshape(X, (X.shape[0], -1))
        # the weight should be [n, out]
        self.weights = np.random.randn(X.shape[1], self.num_output)
        self.bias = np.ones(1) * self.b_val
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
        if self.w_std is None:
            print 'init fc layer weights with default...'
            ns = X.shape[1]
            self.weights *= np.sqrt(2.0 / ns)
        else:
            print 'init fc layer weights with std {}'.format(self.w_std)
            self.weights *= self.w_std

        self.has_init = True


    def forward(self, bottoms):
        """
        forward pass
        :param bottoms:
        :return:
        """
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        assert len(X) == 4 or len(X) == 2
        if len(X) == 4:
            data = np.reshape(X, (X.shape[0], -1))
        else:
            data = X
        check_eq(data.shape[1], self.weights.shape[0], msg='input x not agree with weights')

        # now do the forward job
        out = np.dot(data, self.weights) + self.bias

        # reshape and return
        out = out.reshape((out.shape[0], out.shape[1], 1, 1))
        return out

    def backward(self, tops, propagate_down, bottoms):
        """
        backward pass
        :param tops:
        :param propagate_down:
        :param bottoms:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(bottoms), 1)
        check_eq(len(propagate_down), 1)
        if propagate_down[0]:
            y = tops[0]
            check_eq(len(y), 2)
            X = bottoms[0]
            check_eq(len(X), 2)
            # with respect to w
            self.dw = np.dot(X.T, y)
            # with respect to b
            self.db = np.sum(y)
            # with respect to bottom
            dx = np.dot(y, self.weights.T)
            return dx


