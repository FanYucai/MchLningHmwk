import numpy as np
import matplotlib.pyplot as plt

def addGaussian(num, s, sigma): # mu=0
    mu = 0
    offset_Guassian = np.random.normal(mu, sigma, num)
    s_res = s + offset_Guassian
    return s_res

def genDataInRand(num=20, xrange=3, yrange=1, sigma=0.05):
    interv = xrange*1.0/num
    t = np.arange(0, xrange, interv) #here t is not uniform
    s = yrange*np.sin(2*np.pi*t)
    s = addGaussian(s.shape[0], s, sigma)
    return t, s

def genData(num=20, xrange=3, yrange=1, sigma=0.05):
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
        print 'sum:'
        print sum
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

    print A.shape, B.shape
    weight_res = np.linalg.solve(A, B)
    #>>>>>>>>>>>>>>>>>>>
    x = np.arange(0, 3, 0.01)
    #>>>>>>>>>>>>>>>>>>>

    # print weigh_res.shape
    # print weigh_res

    y = calWithWeight(x, weight_res, max_power)
    return x, y
#
# def minMulPenalty(t, s, max_power):
#     return
#
# def gradiantDescent(t, s, max_power, alpha, iter_times):
#     return
#
# def conjGradiant(t, s, max_power, alpha, iter_times):
#     return
#
# def otherFunc()
#     return

if __name__=="__main__":
    # data_num, data_xr, data_yr, data_thetaoff, data_sigma, data_yoff

    t, s = genData(50)
    plt.plot(t, s, 'go')

    base_t = np.arange(0, 3, 0.001)
    base_s = np.sin(2 * np.pi * base_t)

    plt.axis([0, 3, -1.5, 1.5])
    plt.plot(base_t, base_s, 'r-')
    # plt.plot(np.arange(0, data_xr, 0.01), 5*np.sin(2*np.pi*t+data_theta)+data_yoff, 'g-')

    minMul_t, minMul_s = minMul(t, s, 20)

    plt.plot(minMul_t, minMul_s, 'b-')

    plt.show()


