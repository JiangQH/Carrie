import numpy as np

class MiniBatchGraDescent(object):
    def __init__(self):
        pass

    def fit(self, gfunction, X, y, lr=0.01, max_iter=200, batch_size=5):
        """
        the mini batch gradient descent, perform the mini batch gradient descent
        :param gfunction:
        :param X:
        :param y:
        :param lr:
        :param max_iter:
        :param batch_size:
        :return:
        """
        [m_samples, n_features] = np.shape(X)
        theta = np.random.randn(n_features, 1)
        start = 0
        count = 0
        for iter in range(max_iter):
            end = start + batch_size
            if end > m_samples:
                count += 1
                print 'complete whole {}'.format(count)
                end = end - m_samples - 1
                params = [np.concatenate((X[start:], X[:end])),
                          np.concatenate((y[start:], y[:end])), theta]
            else:
                params = [X[start:end], y[start:end], theta]
            start = end
            theta -= lr * gfunction(*params)

