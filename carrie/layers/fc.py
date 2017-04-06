import numpy as np
from carrie.layers.baselayer import BaseLayer
from carrie.utils.safty_check import check_eq, check_gt

class Fc(BaseLayer):

    """
    this is the fully connected layer
    """
    def __init__(self, name, num_output):
        """
        fully connected layer params
        :param name:
        :param num_output:
        """
        super(Fc, self).__init__(name)
        check_gt(num_output, 1, msg='num output should larger than 1')
        self.num_output = num_output
        self.has_init = False

    def initJob(self, bottoms):
        """
        the init job
        :param bottoms:
        :return:
        """
        if self.has_init:
            return
        check_eq(len(bottoms), 1)
        X = bottoms[0]
        # safety check
        check_eq(len(X.shape), 4)

