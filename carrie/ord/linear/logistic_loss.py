from carrie.ord.layer import Layer
import numpy as np
class LogisticLossLayer(Layer):
    __slots__ = ('_weights', '_gradients')
    def __init__(self, bottom_shape, init_strategy='random'):
        """
        init the weights, which is (n_features, 1), with 0 mean and 1/sqrt(m_samples/2)
        :param bottom_shape:
        :param init_strategy:
        """
        self._weights = np.random.randn(bottom_shape[1], 1) / np.sqrt(bottom_shape[0] / 2)
        self._gradients = np.zeros_like(self._weights)
        super(Layer, self).__init__()


    def forward(self, bottom):
        """
        the forward pass to compute the loss
        -sum(y*log(h(x)) + (1-y)*log(1-h(x)))
        :param bottom:[X, y] X--(m_samples, n_features), y--(m_samples, 1)
        :return:
        """
        assert (len(bottom) == 2)
        X = bottom[0]
        y = bottom[1]
        assert (X.shape[0] == y.shape[0])
        # compute the hypethsis
        h_x = self.__sigmoid(X) # (m_samples, 1)
        return -np.sum(y * np.log(h_x) + (1.0 - y) * np.log(1.0 - h_x))




    def backward(self, top, propagate_down):
        """
        backpropagate the gradient and save the gradient for this layer
        note since this layer is the final layer
        and actually it has the weights, so it
        should bp the loss with respect to the
        input x. which should be multiplied by the weights
        of this layer's gradient
        :param top: this is different, it is the [X, y]
        :param propagate_down: we do not use it
        :return:
        """
        # compute the gradient with respect to Q^t * xe
        # which should be h(x) - y
        assert (top.size() == 2)
        X = top[0]
        y = top[1]
        h_x = self.__sigmoid(X)
        top_diff = (h_x - y) # (m_samples, 1), X-(m_samples, n_features) ---> gradients to w is (n_feateus, 1)
        # now respect to the weights
        self._gradients = np.dot(X.transpose(), top_diff)
        # now back propagate with respect to x
        return np.sum(top_diff) * self._weights

    def __sigmoid(self, bottom):
        """
        compute the sigmoid function 1 / (1 + exp(-Q^t * X))
        :param bottom: (m_samples, n_features)
        :return: (m_samples, 1)
        """
        assert (self._weights.shape[0] == bottom.shape[1])
        return 1.0 / (1 + np.exp(bottom * self._weights))







