#!/usr/bin/python2

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class GradientDescent:

    def __init__(self, x, y, alpha, toggle, iterations):
        self.x = x
        self.y = y

        if toggle == "linreg":
            self.alpha = alpha  # minimising least squares - positive term
        elif toggle == "logreg":
            self.alpha = -alpha  # maximising likelihood estimator - negative term
        elif toggle == "probit":
            self.alpha = alpha  # maximising likelihood estimator - negative term
        self.toggle = toggle
        self.iterations = iterations

        self.m = x.shape[1]
        self.n = x.shape[0]
        self.theta = np.zeros([self.m, 1])
        self.xtrans = x.transpose()
        self.grad = np.zeros([self.m, 1])
        self.cost = np.zeros(self.n)
        self.loss = np.zeros([iterations,1])
        self.hypo = np.zeros(self.n)

    @staticmethod
    def sigmoid(k):
        return 1 / (1 + np.exp(k))

    def hypothesis(self):
        linear = np.dot(self.x, self.theta)
        # Linear case
        if self.toggle == "linreg":
            self.hypo = linear
        elif self.toggle == "logreg":
            self.hypo = self.sigmoid(linear)
        elif self.toggle == "probit":
            self.hypo = norm.cdf(linear)

    def gradient(self):
        if self.toggle == "linreg":
            self.grad = np.dot(self.xtrans, self.cost) / self.n
        elif self.toggle == "logreg":
            self.grad = np.dot(self.xtrans, self.cost) / self.n
        elif self.toggle == "probit":
            linear = np.dot(self.x, self.theta)
            pdf = norm.pdf(linear)
            cdf = norm.cdf(linear)
            f1 = np.divide(pdf, cdf)
            f2 = np.divide(pdf, (1 - cdf))
            self.grad = np.dot(self.xtrans, np.multiply(self.y, f1) - np.multiply(1-self.y, f2)) / self.n

    def batch(self, i):

        # form hypothesis
        self.hypothesis()
        self.cost = self.y - self.hypo
        self.loss[i] = np.sum(self.cost ** 2) / self.n

        # estimate gradient
        self.gradient()

        # update parameter theta
        self.theta = self.theta + self.alpha*self.grad

    def run(self):
        for i in range(0, self.iterations):
            self.batch(i)


def test_linreg():
    # Test Linear regression
    xx = np.ones([100, 2])
    xx[:, 1] = range(0, 100)
    var = 5
    yy = np.dot(xx, [[50], [5]]) + var*np.random.randn(100, 1)

    gd = GradientDescent(xx, yy, 0.0001, "linreg", 10000)

    gd.run()

    f, axarr = plt.subplots(2, sharex=False)
    axarr[0].plot(gd.loss)

    axarr[1].plot(xx[:, 1], yy, '.')
    axarr[1].plot(xx[:, 1], np.dot(xx, gd.theta), '-')
    plt.show()

    print(gd.theta)


def test_logreg():
    # Test Logistic regression
    xx = np.ones([100, 2])
    xx[:, 1] = range(0, 100)
    yy = np.zeros([100, 1])
    yy[70:] = 1

    gd = GradientDescent(xx, yy, 0.01, "logreg", 100000)
    gd.run()

    f, axarr = plt.subplots(2, sharex=False)
    axarr[0].plot(gd.loss)

    axarr[1].plot(xx[:,1], yy, '.')
    axarr[1].plot(xx[:,1], gd.sigmoid(np.dot(xx, gd.theta)), '-')
    axarr[1].plot(xx[:,1], gd.sigmoid(np.dot(xx, gd.theta))>=0.5, '-')

    plt.show()

    print(gd.theta)


def test_probit():
    # Test Logistic regression
    xx = np.ones([100, 2])
    xx[:, 1] = range(0, 100)
    yy = np.zeros([100, 1])
    yy[70:] = 1

    gd = GradientDescent(xx, yy, 0.0001, "probit", 100000)
    gd.run()

    f, axarr = plt.subplots(2, sharex=False)
    axarr[0].plot(gd.loss)

    axarr[1].plot(xx[:,1], yy, '.')
    axarr[1].plot(xx[:,1], gd.sigmoid(np.dot(xx, gd.theta)), '-')
    axarr[1].plot(xx[:,1], gd.sigmoid(np.dot(xx, gd.theta))>=0.5, '-')

    plt.show()

    print(gd.theta)

if __name__ == "__main__":
    # Test the logistic regression
    test_probit()
