import numpy as np

class StoGraDescent(object):
    """
    the stochastic gradient descent
    """
    def __init__(self):
        pass

    def fit(self, gfunction, X, y, lr=0.01, max_eps=100):
        """
        the stochastic gradient descent, note here
        only use on sample each time to perform the update each time
        :param gfunction:
        :param X:
        :param y:
        :param lr:
        :param max_eps:
        :return:
        """
        [m_samples, n_features] = np.shape(X)
        X_spand = np.ones((m_samples, n_features+1))
        theta = np.random.randn(1, n_features+1)
        X_spand[:,1:] = X
        for eps in range(max_eps):
            print "eps {}".format(eps)
            for i in range(m_samples):
                params = [X_spand[i, :], y[i], theta]
                theta += lr * gfunction(*params)
        return theta