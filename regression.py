#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt


class GradientDescent:

    def __init__(self, x, y, alpha, toggle, iterations):
        self.x = x
        self.y = y
        self.alpha = alpha
        self.toggle = toggle
        self.iterations = iterations

        self.m = x.shape[1]
        self.n = x.shape[0]
        self.theta = np.zeros([self.m, 1])
        self.xtrans = x.transpose()
        self.gradient = 0
        self.cost = np.zeros(self.n)
        self.loss = np.zeros([iterations,1])
        self.hypo = np.zeros(self.n)

    def hypothesis(self):
        linear = np.dot(self.x, self.theta)
        # Linear case
        if self.toggle == "linreg":
            self.hypo = linear
        elif self.toggle == "logreg":
            self.hypo = 1 / (1+np.exp(linear))

    def batch(self, i):
        self.hypothesis()
        self.cost = self.y - self.hypo
        self.loss[i] = np.sum(self.cost ** 2)/self.n
        self.gradient = np.dot(self.xtrans, self.cost)/self.n

        # actual algorithm
        self.theta = self.theta + self.alpha*self.gradient

    def run(self):
        for i in range(0, self.iterations):
            self.batch(i)

xx = np.ones([100, 2])
xx[:, 1] = range(0, 100)
var = 5
yy = np.dot(xx, [[50], [5]]) + var*np.random.randn(100, 1)

gd = GradientDescent(xx, yy, 0.0001, "linreg", 100000)

# ls, t = BatchDescent(yy, xx, 0.00001, 100000)
gd.run()

plt.plot(gd.loss)
plt.show()

# plt.plot(yy, '.r')
# plt.plot(np.dot(xx, t), '.b')
# plt.show()

print(gd.theta)