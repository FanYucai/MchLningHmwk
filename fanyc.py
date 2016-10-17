import numpy as np
import matplotlib.pyplot as plt

def addGaussian(num, s, sigma): # mu=0
    mu = 0
    offset_Guassian = np.random.normal(mu, sigma, num)
    s_res = s + offset_Guassian
    return s_res

def genDataInRand(num=10, xrange=2, yrange=1, sigma=0.05):
    interv = xrange*1.0/num
    t = np.arange(0, xrange, interv) #here t is not uniform
    s = yrange*np.sin(2*np.pi*t)
    s = addGaussian(s.shape[0], s, sigma)
    return t, s

def genData(num=10, xrange=2, yrange=1, sigma=0.05):
    interv = xrange*1.0/num
    t = np.arange(0, xrange, interv)
    s = yrange*np.sin(2*np.pi*t)
    s = addGaussian(s.shape[0], s, sigma)
    return t, s

def calWithWeigh(x, weigh):
    order = weigh.shape[0]
    return y

def minMul(t, s, max_power):
    A = []
    for i in range(0, max_power+1):
        A_sub = []
        for j in range(0, max_power+1):
            A_sub.append(np.power(t[i], j))
        A.append(A_sub)
    mat_A = np.mat(A)
    # print A

    mat_B = np.transpose(np.mat(s))
    # print B

    weigh_res = np.linalg.solve(mat_A, mat_B)

    # x = np.arange(0, xrange, 0.01)
    x = np.arange(0, 2, 0.01)
    y = calWithWeigh(x, weigh_res)
    return x
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

    t, s = genData(20)
    plt.plot(t, s, 'go')

    base_t = np.arange(0, 2, 0.001)
    base_s = np.sin(2 * np.pi * base_t)

    plt.axis([0, 2, -1.5, 1.5])
    plt.plot(base_t, base_s, 'r-')
    # plt.plot(np.arange(0, data_xr, 0.01), 5*np.sin(2*np.pi*t+data_theta)+data_yoff, 'g-')

    minMul(t, s, 6)


    plt.show()


