class Layer(object):
    """
    the base class for layers
    """
    def __init__(self):
        pass

    def forward(self, bottom):
        pass

    def backward(self, top, propagate_down):
        pass
