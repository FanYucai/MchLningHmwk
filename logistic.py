# coding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt

def addGaussian(num, s, sigma): # mu=0
    mu = 0
    offset_Guassian = np.random.normal(mu, sigma, num)
    s_res = s + offset_Guassian
    return s_res

def genData(num=20, xrange=1, yrange=1, sigma=0.1):
    interv = xrange*1.0/num
    t = np.arange(0, xrange, interv)
    s = yrange*np.sin(2*np.pi*t)
    s = addGaussian(s.shape[0], s, sigma)
    return t, s

def calWithWeight(x, weight, max_power):
    y = []
    for iter_x in x:
        sum = 0.0
        for i in range(0, max_power+1):
            sum += 1.0*np.power(iter_x, i)*weight[i]
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

    weight_res = np.linalg.solve(A, B)
    x = np.arange(0, 3, 0.01)
    print 'weight_res:'
    print weight_res
    y = calWithWeight(x, weight_res, max_power)
    # return x, y
    return x, y

def calSingleCha(xi, yi, weight, max_power): #weight.shape = power+1
    res = 0
    for w in range(0, max_power+1):
        res += np.power(xi, w)*weight[w]
    res -= yi
    return res

def calJ(x, y, weight, max_power):
    J = 0
    for i in range(0, x.shape[0]):
        xi = x[i]
        yi = y[i]
        J += np.power(calSingleCha(xi, yi, weight, max_power), 2)
    return J

def gradientDescent(t, s, max_power, alpha=0.5, iter_times=20000):
    #初始化权重
    weight = 200*np.random.random(size=max_power+1)-100
    epsilon = 1e-6
    cuJ = 0
    preJ = calJ(t, s, weight, max_power)
    num = len(t)

    while np.abs(preJ-cuJ)>epsilon and iter_times>0:
        iter_times -= 1
        preJ = cuJ
        for j in range(0, max_power+1):
            sum = 0
            for i in range(0, num):
                sum += np.power(t[i], j)*(calSingleCha(t[i], s[i], weight, max_power))
            weight[j] = weight[j] - alpha*sum
        cuJ = calJ(t, s, weight, max_power)
        print iter_times, ' :', cuJ

    if iter_times < 0 :
        print "Exhausted..try again"

    x = np.arange(0, 3, 0.01)
    y = calWithWeight(x, weight, max_power)
    return x, y

def Hessian(t, max_power):
    step_dd = []
    num = t.shape[0]
    thetaNum = max_power+1
    for row in range(0, thetaNum):
        step_dd_row = []
        for col in range(0, thetaNum):
            sum = 0
            for j in range(0, num):
                sum += np.power(t[j], row+col)
            step_dd_row.append(sum)
        step_dd.append(step_dd_row)
    # print step_dd
    return np.array(step_dd)

def Jacobian(t, s, weight, max_power):
    # delta J(theta) Jacobian
    step_d = []
    num = t.shape[0]
    thetaNum = max_power+1
    for j in range(0, thetaNum):
        sum = 0
        for i in range(0, num):
            sum += np.power(t[i], j)*(calSingleCha(t[i], s[i], weight, max_power))
        step_d.append(sum)
    return np.array(step_d)


if __name__=="__main__":
    base_t = np.arange(0, 1, 0.001)
    base_s = np.sin(2 * np.pi * base_t)
    t, s = genData(20)

    plt.axis([0, 1, -2, 2])
    plt.plot(t, s, 'g.')
    plt.plot(base_t, base_s, 'pink')
    minMul_t, minMul_s = minMul(t, s, 7)
    plt.plot(minMul_t, minMul_s, 'black')
    plt.title('Least Square')

    plt.show()


