from carrie.baselayer import BaseLayer

class Convolution(BaseLayer):
    """
    this is the convolution layer
    """
    def __init__(self, kernel_width = 3, kernel_height = 3, kernel_num = 64,
                 pad = 1, stride = 1):
        """
        convolution layer params
        :param kernel_width: the kernel width
        :param kernel_height: the kernel height
        :param kernel_num: kernel nums
        :param pad: padding to the input
        :param stride: the stride of kernels
        """
        assert kernel_width > 0 and kernel_height > 0
        assert kernel_num > 0 and pad > 1 and stride > 1
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_num = kernel_num
        self.pad = pad
        self.stride = stride


    def forward(self, X):
        """
        compute the 
        :param X:
        :return:
        """