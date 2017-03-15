class BaseLayer(object):

    def forward(self, X):
        """
        :param X: forward step, input. which is a tensor [n, c, w, h]
        :return: the forward results
        """
        pass

    def backward(self, Y):
        """
        :param Y: backward step, output. which is a tensor [n, c, w, h]
        :return: back_diff
        """
        pass