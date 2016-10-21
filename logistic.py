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
    if  np.sin(2*np.pi*Dx) < Dy:
        return 1
    else:
        return 0
def genData(num=20):
    # s = addGaussian(s.shape[0], s, sigma)
    Dx = np.random.random(size=num)
    Dy = 4*np.random.random(size=num)-2

    # negativeDx = np.random.random(size=20)
    return Dx, Dy

def calWiththeta(x, theta):
    y = []
    for iter_x in x:
        sum = 0.0
        for i in range(0, 3):
            sum += np.power(iter_x, i)*theta[i]
        y.append(sum)
    return y

def minMul(t, s, max_power):
    A = []
    size = t.shape[0]
    print size

    for k in range(0, max_power+1):
        A_sub = []
        for j in range(0, max_power+1):
            sum_t = 0
            for i in range(0, size):
                sum_t += np.power(t[i], j+k)
            A_sub.append(sum_t)
        A.append(A_sub)
    A = np.array(A)

    B = []
    for k in range(0, max_power+1):
        sum_s = 0
        for i in range(0, size):
            sum_s += np.power(t[i], k)*s[i]
        B.append(sum_s)
    B = np.array(B)

    theta_res = np.linalg.solve(A, B)
    x = np.arange(0, 3, 0.01)
    print 'theta_res:'
    print theta_res
    y = calWiththeta(x, theta_res, max_power)
    # return x, y
    return x, y

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

def gradientDescent(t, s, alpha=0.01, iter_times=20000):
    theta = 20*np.random.random(size=3)-10
    print theta
    epsilon = 1e-6
    cuJ = 0
    preJ = calJ(t, s, theta)
    num = len(t)
    # thetaNum = len(theta)
    thetaNum = 3

    # while np.abs(preJ-cuJ)>epsilon and iter_times>0:
    while iter_times>0:
        iter_times -= 1
        preJ = cuJ
        for j in range(0, thetaNum):
            sum = 0
            for i in range(0, num):
                if j==0:
                    # print sigmoid(h(t[i], s[i], theta)), theta
                    sum += sigmoid(h(t[i], s[i], theta))-judgeD(t[i], s[i])
                else:
                    sum += t[i]*(sigmoid(h(t[i], s[i], theta))-judgeD(t[i], s[i]))

            theta[j] = theta[j] - alpha*sum
        cuJ = calJ(t, s, theta)
        print iter_times, ':', cuJ

    if iter_times <= 0 :
        print "Exhausted..try again"

    x = np.arange(0, 3, 0.01)
    y = calWiththeta(x, theta)
    return x, y

if __name__=="__main__":
    base_t = np.arange(0, 1, 0.001)
    base_s = np.sin(2 * np.pi * base_t)
    Dx, Dy = genData(30)

    plt.figure(1)

    plt.axis([0, 1, -2, 2])
    for i in range(len(Dx)):
        dx = Dx[i]
        dy = Dy[i]
        if judgeD(dx, dy) == 1:
            plt.plot(dx, dy, 'bo')
        else:
            plt.plot(dx, dy, 'ro')

    plt.plot(base_t, base_s, 'pink')
    grad_t, grad_s = gradientDescent(Dx, Dy, 7)
    plt.plot(grad_t, grad_s, 'g')

    plt.show()


