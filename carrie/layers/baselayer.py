class BaseLayer(object):

    def __init__(self, name):
        self._name = name


    def forward(self, bottoms):
        """
        :param bottoms: bottoms, which is a list of tensors of [n, c, h, w]
        :return: the forward results
        """
        pass

    def backward(self, tops, propagate_down, bottoms):
        """
        backward pass
        :param tops: list of top tensors
        :param propagate_down: will we propagate down to the bottom_i ?
        :param bottoms: list of bottom tensors, data, which may be needed when do update to the weights
        :return:
        """
        pass

    def initJob(self, bottoms):
        """
        the bottoms data, do some init job here
        :param bottoms:
        :return:
        """
        pass