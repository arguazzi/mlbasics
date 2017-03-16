#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Author: Alessandro Guazzi

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class GradientDescent:

    def __init__(self, x, y, alpha, iterations):
        self.x = x
        self.y = y
        self.alpha = alpha
        self.iterations = iterations

        self.m = x.shape[1]
        self.n = x.shape[0]
        self.theta = np.zeros([self.m, 1])
        self.xtrans = x.transpose()
        self.grad = np.zeros([self.m, 1])
        self.cost = np.zeros(self.n)
        self.loss = np.zeros([iterations, 1])
        self.hypo = np.zeros(self.n)

    def calculate_hypothesis(self):
        self.hypo = np.dot(self.x, self.theta)

    def calculate_gradient(self):
        self.grad = np.dot(self.xtrans, self.cost) / self.n

    def calculate_cost(self):
        self.cost = self.hypo - self.y

    def update_theta(self):
        self.theta = self.theta - self.alpha * self.grad

    def batch(self, i):
        self.calculate_hypothesis()
        self.calculate_cost()
        self.loss[i] = np.sum(self.cost ** 2) / self.n
        self.calculate_gradient()
        self.update_theta()

    def run(self):
        for i in range(self.iterations):
            self.batch(i)


class LinearRegression(GradientDescent):
    pass


class LogisticRegression(GradientDescent):

    @staticmethod
    def sigmoid(k):
        return 1 / (1 + np.exp(k))

    def calculate_hypothesis(self):
        # Correct the hypothesis to lie between 0 and 1
        self.hypo = self.sigmoid(np.dot(self.x, self.theta))

    def update_theta(self):
        # Correct to a positive as are using gradient ascent (MLE)
        self.theta = self.theta + self.alpha * self.grad


class ProbitRegression(GradientDescent):

    def calculate_hypothesis(self):
        # Probit model takes the CDF as the function to normalise between 0 and 1
        self.hypo = norm.cdf(np.dot(self.x, self.theta))

    def calculate_gradient(self):
        # Gradient does not simplify much in the case of the Probit model
        linear = np.dot(self.x, self.theta)
        pdf = norm.pdf(linear)
        cdf = norm.cdf(linear)
        f1 = np.divide(pdf, cdf)
        f2 = np.divide(pdf, (1 - cdf))
        self.grad = np.dot(self.xtrans, np.multiply(self.y, f1) - np.multiply(1 - self.y, f2)) / self.n

    def update_theta(self):
        # Correct for gradient ascent
        self.theta = self.theta + self.alpha * self.grad


def test(family):
    if family == 'linear':
        xx = np.ones([100, 2])
        xx[:, 1] = range(0, 100)
        var = 5
        yy = np.dot(xx, [[50], [5]]) + var * np.random.randn(100, 1)
    elif (family == 'logistic') | (family == 'probit'):
        xx = np.ones([100, 2])
        xx[:, 1] = range(0, 100)
        yy = np.zeros([100, 1])
        yy[70:] = 1
        yy[77] = 0

    if family == 'linear':
        gd = LinearRegression(xx, yy, 0.0001, 10000)
        gd.run()
    elif family == 'logistic':
        gd = LogisticRegression(xx, yy, 0.001, 100000)
        gd.run()
    elif family == 'probit':
        gd = ProbitRegression(xx, yy, 0.001, 100000)
        gd.run()

    if family == 'linear':
        f, axarr = plt.subplots(2, sharex=False)
        axarr[0].plot(gd.loss)

        axarr[1].plot(xx[:, 1], yy, '.')
        axarr[1].plot(xx[:, 1], np.dot(xx, gd.theta), '-')

    elif (family == 'logistic') | (family == 'probit'):
        f, axarr = plt.subplots(2, sharex=False)
        axarr[0].plot(gd.loss)

        axarr[1].plot(xx[:, 1], yy, '.')
        axarr[1].plot(xx[:, 1], norm.cdf(np.dot(xx, gd.theta)), '-')
        axarr[1].plot(xx[:, 1], norm.cdf(np.dot(xx, gd.theta)) >= 0.5, '-')

    plt.show()

if __name__ == "__main__":
    # Test the logistic regression
    test('logistic')
