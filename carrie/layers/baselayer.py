class BaseLayer(object):

    def __init__(self, name):
        self._name = name


    def forward(self, X, y):
        """
        :param X: forward step, input. which is a tensor [n, c, w, h]
        :return: the forward results
        """
        pass

    def backward(self, y, X):
        """
        :param Y: backward step, output. which is a tensor [n, c, w, h]
        :return: back_diff
        """
        pass

    def initJob(self, X):
        """
        init the weights and bias, call only once
        :param X:
        :return:
        """