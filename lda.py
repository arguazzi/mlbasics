#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt


class LDA:

    def __init__(self):
        self.sigma_inv = 0
        self.const = 0
        self.coeff = 0

    def train(self, train_input, train_class):

        # Input matrix with values as columns and variables as rows.
        x = train_input
        y = train_class
        m = x.shape[0]
        n = x.shape[1]
        k = len(np.unique(train_class))
        self.const = np.zeros(k)
        self.coeff = np.zeros([m, k])
        sigma = np.cov(x)

        sigma_inv = np.linalg.inv(sigma)

        for i in range(0, k):
            xx = x[:, y == i]
            p = xx.shape[0]/n
            mu = xx.mean(axis=1)
            mut = mu.transpose()
            self.const[i] = np.dot(mut, np.dot(sigma_inv, mu)) + np.log(p)
            self.coeff[:, i] = np.dot(sigma_inv, mu)

    def test(self, test_input):

        x_t = test_input.transpose()

        d = self.const - 2*np.dot(x_t, self.coeff)

        return d

if __name__ == '__main__':
    # Test that it actually works
    A = np.random.multivariate_normal([1, 2], [[0.3, 0.1], [0.2, 0.3]], 100)*1.0
    B = np.random.multivariate_normal([2, 4], [[0.3, 0.2], [0.2, 0.2]], 100)*1.0
    C = np.random.multivariate_normal([0, 0], [[0.3, 0.22], [0.2, 0.2]], 100)*1.0
    X = np.concatenate([A, B, C], axis=0).transpose()
    Y = np.concatenate([np.zeros(100), np.ones(100), np.ones(100)*2.0])

    plt.scatter(A[:, 1], A[:, 0], marker='*')
    plt.scatter(B[:, 1], B[:, 0], marker='o')
    plt.scatter(C[:, 1], C[:, 0], marker='s')
    plt.show()

    l = LDA()
    l.train(X, Y)
    d = l.test(X)

    plt.scatter(d[d[:, 0] > d[:, 1], 1], d[d[:, 0] > d[:, 1], 2], marker='x')
    plt.scatter(d[d[:, 0] < d[:, 1], 1], d[d[:, 0] < d[:, 1], 2], marker='s')
    plt.show()










