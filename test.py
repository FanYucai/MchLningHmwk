# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import random as rand

if __name__=="__main__":
    dataNum = 30
    mean = [0, 0]
    cov = [[100, 60], [1, 10]]
    x, y = np.random.multivariate_normal(mean, cov, 300).T
    plt.plot(x, y, 'x')
    plt.axis('equal')

    plt.show()


