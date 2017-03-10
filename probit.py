#!/usr/bin/python2

# probit model for binary classification
#
# For derivations of the gradient look at:
# https://pdfs.semanticscholar.org/0e3b/1a44588b8f5c3140a8730143319a8063c4b2.pdf
# http://www.yindawei.com/wp-content/uploads/2012/11/probitRegression.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Stochastic ascent - does not seem to work
def StochasticAscent(y, x, alpha, iterations):
    m = x.shape[1]
    n = x.shape[0]
    theta = np.zeros(m)
    Xtran = x.transpose()
    cost = np.zeros(x.shape[0]*iterations+1)

    count = 0

    for j in range(0, iterations):

        for i in range(0, x.shape[0]):
            linear = np.dot(theta, Xtran[:,i])
            pdf = norm.pdf(linear)
            cdf = norm.cdf(linear)
            f1 = pdf/cdf
            f2 = pdf/(1-cdf)

            cost[count] = np.sum((y[i] - cdf)**2)/n

            gradient = (y[i]*f1) - (1-y[i])*f2

            theta = theta + alpha * gradient/n

            count += 1

    return cost, theta

# Batch ascent - does seem to work quite well
def BatchAscent(y, x, alpha, iterations):
    m = x.shape[1]
    n = x.shape[0]
    theta = np.zeros([m, 1])
    xtrans = x.transpose()
    cost = np.zeros(iterations)

    for i in range(0, iterations):
        linear = np.dot(x, theta)
        pdf = norm.pdf(linear)
        cdf = norm.cdf(linear)
        f1 = np.divide(pdf, cdf)
        f2 = np.divide(pdf, (1-cdf))
        loss = y - cdf
        cost[i] = np.sum(loss**2)/n

        gradient = np.dot(xtrans, np.multiply(y, f1) - np.multiply(1-y, f2))

        theta = theta + alpha * gradient/n

    return cost, theta


# Test binary regression
xx = np.ones([100, 2])
xx[:, 1] = range(0, 100)
yy = np.zeros([100, 1])
yy[70:] = 1

c, t = BatchAscent(yy, xx, 0.001, 1000)

f, axarr = plt.subplots(2, sharex=False)
axarr[0].plot(c)

axarr[1].plot(xx[:, 1], yy, '.')
axarr[1].plot(xx[:, 1], norm.cdf(np.dot(xx, t)), '-')
axarr[1].plot(xx[:, 1], norm.cdf(np.dot(xx, t)) >= 0.5, '-')

plt.show()
