import numpy as np
"""
abstract the im2col and col2im operation in here,
since it can be used by both convolution layer and deconv layer
"""


def __get_im2col_index(x_shape, kernel_height, kernel_width, pad, stride):
    N, C, H, W = x_shape
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

    # note we should return the index of after padded , so remember it
    w_kernel = np.repeat(np.arange(kernel_height), kernel_width)
    w_kernel = np.tile(w_kernel, C)
    adder_w = stride * np.repeat(np.arange(out_height), out_width)
    w_index = w_kernel.reshape(-1, 1) + adder_w.reshape(1, -1)

    h_kernel = np.tile(np.arange(kernel_width), kernel_height * C)
    adder_h = stride * np.tile(np.arange(out_width), out_height)
    h_index = h_kernel.reshape(-1, 1) + adder_h.reshape(1, -1)

    k = np.repeat(np.arange(C), kernel_height * kernel_width).reshape(-1, 1)

    return (k.astype(int), w_index.astype(int), h_index.astype(int))



def im2col(X, kernel_height, kernel_width, pad, stride):
    """
    do the im2col job, get the val which we want to output
    :param X:
    :param kernel_height:
    :param kernel_width:
    :param pad:
    :param stride:
    :return:
    """
    x_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    ch_index, h_index, w_index = __get_im2col_index(X.shape, kernel_height, kernel_width, pad, stride)

    c = X.shape[1]
    col = x_padded[:, ch_index, h_index, w_index]
    col = col.transpose(1, 2, 0).reshape(c * kernel_height * kernel_width, -1)
    return col

def col2im(cols, X_shape, kernel_height, kernel_width, pad, stride):
    """
    do the col2im job, this can be used when do backprop
    :param cols:
    :param X:
    :param kernel_height:
    :param kernel_width:
    :param pad:
    :param stride:
    :return:
    """
    [N, C, H, W] = X_shape
    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    ch_index, h_index, w_index = __get_im2col_index(X_shape, kernel_height, kernel_width, pad, stride)
    cols_r = cols.reshape(C * kernel_height * kernel_width, -1).transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), ch_index, h_index, w_index), cols_r)

    return x_padded if pad == 0 else x_padded[:, :, pad:-pad, pad:-pad]
