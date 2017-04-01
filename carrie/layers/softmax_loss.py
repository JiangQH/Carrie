from carrie.layers.baselayer import BaseLayer

class SoftmaxLossLayer(BaseLayer):
    """
    the softmax loss layer, using the softmax function behind
    """
    def __init__(self, name):
        super(SoftmaxLossLayer, self).__init__(name)

    def forward(self, bottoms):
        """
        :param bottoms:
        :return:
        """
        pass


    def backward(self, tops, propagate_down, bottoms):
        """
        :param tops:
        :param propagate_down:
        :param bottoms:
        :return:
        """
        pass