# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(s):
    return 1.0/(1+np.power(np.e, -s))

def addGaussian(num, s, sigma): # mu=0
    mu = 0
    offset_Guassian = np.random.normal(mu, sigma, num)
    s_res = s + offset_Guassian
    return s_res

def judgeD(Dx, Dy):
    if  np.sin(1*np.pi*Dx) < Dy:
        return 1
    else:
        return 0
def genData(num=20):
    Dx = 1.4*np.random.random(size=num)+0.3
    Dy = 4*np.random.random(size=num)-2
    dataMat = []
    labelMat = []
    for i in range(num):
        dataMat.append([1.0, Dx[i], Dy[i]])
        labelMat.append(judgeD(Dx[i], Dy[i]))
    return dataMat, labelMat

def calWiththeta(x, theta):
    y = []
    theta = np.array(theta)
    for xi in x:
        yi = (-theta[0]-theta[1]*xi)/theta[2]
        y.append(yi)
    return y

def h(x1, x2, theta):
    return theta[0]+theta[1]*x1+theta[2]*x2

def calJ(x1, x2, theta):
    J = 0
    for i in range(0, x1.shape[0]):
        x1i = x1[i]
        x2i = x2[i]
        yi = judgeD(x1i, x2i)
        J += yi*np.log(sigmoid(h(x1i, x2i, theta)))
        J += (1-yi)*np.log(1-h(x1i, x2i, theta))
    return -J

def gradientDescent(dataMatIn, labelMatIn, dataNum, alpha=0.001, iter_times=10000):
    # theta = 10*np.random.random(size=3)-10
    theta = np.ones((3, 1))
    print theta
    dataMat = np.mat(dataMatIn)
    print dataMat
    labelMat = np.mat(labelMatIn).transpose()

    for iter in range(iter_times):
        afterSig = sigmoid(dataMat*theta)
        theta = theta - alpha * dataMat.transpose()*(afterSig-labelMat)

    print 'theta :', theta

    x = np.arange(0, 3, 0.01)

    y = calWiththeta(x, theta)

    return x, y

if __name__=="__main__":
    # base_t = np.arange(-10, 10, 0.001)
    dataNum = 30
    base_t = np.arange(0, 3, 0.001)
    base_s = np.sin(1 * np.pi * base_t)
    dataMat, labelMat = genData(dataNum)
    # plt.plot(base_t, sigmoid(base_t))

    plt.figure(1)

    plt.axis([0, 2, -2.5, 2.5])
    for i in range(dataNum):
        if labelMat[i] == 1:
            plt.plot(dataMat[i][1], dataMat[i][2], 'bo')
        else:
            plt.plot(dataMat[i][1], dataMat[i][2], 'ro')

    plt.plot(base_t, base_s, 'pink')
    grad_t, grad_s = gradientDescent(dataMat, labelMat, dataNum)
    plt.plot(grad_t, grad_s, 'g')

    plt.show()


