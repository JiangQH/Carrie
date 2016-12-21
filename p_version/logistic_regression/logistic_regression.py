import numpy as np
from optimization.stochastic_gradient_descent import StoGraDescent as SGD
class LogisticRegression(object):
    def __init__(self):
        __slots__ = 'theta'

    def fit(self, X, y):
        """
        fit the logistic regression
        :param X: input X which is (m_samples, n_features)
        :param y: input y label, which is(m_samples,)
        :return: nothing, store the params in object
        """
        # expand the X
        [m_samples, n_features] = np.shape(X)
        assert m_samples == len(y), "X and y must have same number"
        X_spand = np.ones((m_samples, n_features+1))
        X_spand[:, 1:] = X
        # train the model using gradient descent
        # here use the stochastic_gradient_descent
        self.theta = SGD().fit(self.lossGrad, X, y, lr=0.01, max_eps=100)

    def predict(self, test_X):
        """
        predict the label according to test_X and self.theta
        :param test_X:
        :return:
        """
        [m_samples, n_features] = np.shape(test_X)
        assert (n_features+1) == len(self.theta), "test_X feature size not agree with the model"
        predition = np.zeros((m_samples, 1), dtype=np.int32)
        h_x = self.sigmoid(test_X * self.theta)
        predition[h_x > 0.5] = 1
        return predition



    def sigmoid(self, Z):
        """
        the sigmoid function
        :param Z: the input Z
        :return: sigmoid function value
        """
        return 1 / (1 + np.exp(-Z))

    def loss(self, X, y):
        """
        the loss function of logistic regression
        :param X: (m_samples, n_features)
        :param y:
        :return:
        """
        h_x = self.sigmoid(X * self.theta)
        return np.transpose(y) * np.log(h_x) + np.transpose(1 - y) * np.log(1 - h_x)

    def lossGrad(self, params):
        X = params[0]
        y = params[1]
        theta = params[2]
        h_x = self.sigmoid(X * theta)
        return np.transpose(X) * (h_x - y)