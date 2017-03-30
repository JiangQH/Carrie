import numpy as np
"""
abstract the im2col and col2im operation in here,
since it can be used by both convolution layer and deconv layer
"""


def __get_im2col_index(X, kernel_height, kernel_width, pad, stride):
    [N, C, H, W] = X.shape
    # safety check
    assert (H + 2 * pad - kernel_height) % stride == 0, 'kernel_height and input height not agree. {} vs {}'.format(
        kernel_height, H
    )
    assert (W + 2 * pad - kernel_width) % stride == 0, 'kernel_width and input width not agree. {} vs {}'.format(
        kernel_width, H
    )
    # get the output size
    out_height = (H + 2 * pad - kernel_height) / stride + 1
    out_width = (W + 2 * pad - kernel_width) / stride + 1
    # get the index


def im2col(X, kernel_height, kernel_width, pad, stride):
    pass

def col2im(cols, X, kernel_height, kernel_width, pad, stride):
    pass
