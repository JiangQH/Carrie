import numpy as np

class BathGraDescent(object):
    """
    this file implements the batch gradient descent algorithm
    need the user provide the gradient function
    """
    def __init__(self):
        pass

    def fit(self, gfunction, X, y, lr=0.01, max_iter=200):
        """
        using batch gradient descent to optimize.
        note that we add a x0=1 to the data, so the theta_0 bias can be
        included in the params
        :param gfunction: the user provided gradient function. it should be able to perform
                vector operation. it receives a list as [X, y]
        :param X: the training data, which is (m_samples, n_features)
        :param y: the corresponding label, which is (m_samples, 1)
        :return: theta, which is (n_features, 1)
        """
        [m_samples, n_features] = np.shape(X)
        # the theta
        theta = np.random.randn(n_features, 1)
        # do the update
        for iter in range(max_iter):
            params = [X, y, theta]
            theta -= lr * gfunction(*params)

        return theta



