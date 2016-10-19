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

def gradientDescent(t, s, max_power, alpha=0.1, iter_times=10000):
    #初始化权重
    weight = 200*np.random.random(size=max_power+1)-100
    epsilon = 1e-5
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


def gradientDescentP (t, s, max_power, alpha=0.1, iter_times=10000):
    #初始化权重
    weight = 200*np.random.random(size=max_power+1)-100
    epsilon = 1e-5
    cuJ = 0
    preJ = calJ(t, s, weight, max_power)
    num = len(t)
    lam = 0.1

    while np.abs(preJ-cuJ)>epsilon and iter_times>0:
        iter_times -= 1
        preJ = cuJ
        for j in range(0, max_power+1):
            sum = 0
            for i in range(0, num):
                sum += np.power(t[i], j)*(calSingleCha(t[i], s[i], weight, max_power))
            weight[j] = weight[j] - alpha*(sum+lam*weight[j])
        cuJ = calJ(t, s, weight, max_power)
        print iter_times, ':', cuJ

    if iter_times < 0 :
        print "Exhausted..try again"

    x = np.arange(0, 3, 0.01)
    y = calWithWeight(x, weight, max_power)
    return x, y

def getNorm(g):
    res = 0
    for i in g:
        res += i**2
    return res

def ConjGrad(t, s, weight, max_power):
    r = -Jacobian(t, s, weight, max_power)
    p=r #futidu

    betaTop = np.dot(r.transpose(),r)
    beta0 = betaTop

    i, k = 0, 0
    iter_fir = 10000
    epsilon = 1e-6
    iter_sec = 100

    nRestart = np.shape(weight)[0]
    w = weight
    print "w init:", w

    while i<iter_fir and betaTop > epsilon**2*beta0:
        j=0
        dp = np.dot(p.transpose(),p)
        alpha = (epsilon+1)**2

        # Newton-Raphson ,在该搜索方向上移动到极小
        while j<iter_sec and alpha**2 * dp > epsilon**2:
            alpha = -np.dot(Jacobian(t, s, w, max_power).transpose(),p) / (np.dot(p.transpose(),np.dot(Hessian(t, max_power),p)))
            print "alpha:", alpha
            w = w + alpha * p
            j += 1
        print 'w: ', w
        print '------'

        #计算beta
        r = -Jacobian(t, s, w, max_power)
        betaBottom = betaTop
        betaTop = np.dot(r.transpose(),r)
        beta = betaTop/betaBottom

        #更新p
        p = r + beta*p
        k += 1

        #重新开始
        if k==nRestart or np.dot(r.transpose(),p) <= 0:
            p = r
            k = 0
        i +=1
    return w

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
    print step_dd
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

def conjGradient(t, s, max_power):
    weight = 200*np.random.random(size=max_power+1)-100
    x = np.arange(0, 3, 0.01)
    weight = ConjGrad(t, s, weight, max_power)
    y = calWithWeight(x, weight, max_power)
    return x, y

if __name__=="__main__":
    # data_num, data_xr, data_yr, data_thetaoff, data_sigma, data_yoff
    base_t = np.arange(0, 1, 0.001)
    base_s = np.sin(2 * np.pi * base_t)
    t, s = genData(50)

    plt.figure(1)

    plt.subplot(221)
    plt.axis([0, 1, -2, 2])
    plt.plot(t, s, 'g.')
    plt.plot(base_t, base_s, 'black')
    minMul_t, minMul_s = minMul(t, s, 7)
    plt.plot(minMul_t, minMul_s, 'r')
    plt.title('Least Square')

    plt.subplot(222)
    plt.axis([0, 1, -2, 2])
    plt.plot(t, s, 'g.')
    plt.plot(base_t, base_s, 'black')
    # grad_t, grad_s = gradientDescent(t, s, 9, minMul_weight)
    grad_t, grad_s = gradientDescent(t, s, 7)
    plt.plot(grad_t, grad_s, 'b')
    plt.title('Grad Descent(without penalty)')

    plt.subplot(223)
    plt.axis([0, 1, -2, 2])
    plt.plot(t, s, 'g.')
    plt.plot(base_t, base_s, 'black')
    conj_t, conj_s = conjGradient(t, s, 7)
    plt.plot(conj_t, conj_s, 'r')
    plt.title('Conj Gradient')

    plt.subplot(224)
    plt.axis([0, 1, -2, 2])
    plt.plot(t, s, 'g.')
    plt.plot(base_t, base_s, 'black')
    gradP_t, gradP_s = gradientDescentP(t, s, 7)
    plt.plot(gradP_t, gradP_s, 'b')
    plt.title('Gradient Descent with Penalty')

    plt.show()


