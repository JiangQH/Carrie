from carrie.layers.baselayer import BaseLayer
from carrie.utils.safty_check import check_eq
import numpy as np
class EuclideanLoss(BaseLayer):

    def __init__(self, name):
        super(EuclideanLoss, self).__init__(name)


    def forward(self, bottoms):
        """
        forward pass, compute the euclidean loss
        :param bottoms:
        :return:
        """
        check_eq(len(bottoms), 2)
        logits = bottoms[0]
        labels = bottoms[1]
        check_eq(logits.shape, labels.shape)

        m = logits.shape[0]
        loss = 0.5 * np.sum((logits - labels)**2) / m
        return loss


    def backward(self, tops, propagate_down, bottoms):
        """
        backward pass
        :param tops:
        :param propagate_down:
        :param bottoms:
        :return:
        """
        check_eq(len(tops), 1)
        check_eq(len(propagate_down), 2)
        check_eq(len(bottoms), 2)
        logits = bottoms[0]
        labels = bottoms[1]
        check_eq(logits.shape, labels.shape)
        if propagate_down[0]:
            grad_y = logits - labels
            m = logits[0]
            grad_y /= m
            return grad_y