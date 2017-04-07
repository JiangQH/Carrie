from carrie.layers.baselayer import BaseLayer
from carrie.utils.safty_check import check_eq
from carrie.math.math import softmax
import numpy as np
class SoftmaxLossLayer(BaseLayer):
    """
    the softmax loss layer, using the softmax function behind
    """
    def __init__(self, name):
        super(SoftmaxLossLayer, self).__init__(name)

    def forward(self, bottoms):
        """
        :param bottoms: it should have two bottoms, 1st is the prediction, 2nd is the real label one
        :return:
        """
        check_eq(len(bottoms), 2)
        logits = bottoms[0]
        labels = bottoms[1]
        check_eq(len(logits), 2)
        check_eq(len(labels), 2)
        check_eq(labels.shape[1], 1)
        check_eq(logits.shape[0], labels.shape[0])
        # the behind softmax loss
        prob = softmax(logits)
        log_like = -np.log(prob[prob[:, labels]])

        m = labels.shape[0]
        data_loss = np.sum(log_like) / m
        # do the regularization outside
        return data_loss



    def backward(self, tops, propagate_down, bottoms):
        """
        :param tops:
        :param propagate_down:
        :param bottoms:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(bottoms), 2)
        logits = bottoms[0]
        labels = bottoms[1]
        check_eq(len(logits), 2)
        check_eq(len(labels), 2)
        check_eq(labels.shape[1], 1)
        check_eq(logits.shape[0], labels.shape[0])
        # the softmax function
        if propagate_down[0]:
            grad_y = softmax(logits)
            grad_y[:, labels] -= 1
            m = labels.shape[0]
            grad_y /= m
            return grad_y
        