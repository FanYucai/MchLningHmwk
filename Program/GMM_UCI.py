# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def phi(x, mu, cov):

    x = np.matrix(x)
    mu = np.matrix(mu)
    cov = np.matrix(cov)

    x_mu = x-mu
    # print cov
    # s = -1.0/2 * x_mu * np.abs(np.linalg.inv(cov)) * x_mu.T

    if(np.linalg.det(cov) > 0):
        s = -1.0/2 * x_mu * np.linalg.inv(cov) * x_mu.T
        res = 1.0/((2*np.pi) * np.sqrt(np.linalg.det(cov))) * (np.power(np.e, s))
    elif(np.linalg.det(cov) == 0):
        cov += np.matrix([[np.random.random()*1e-6, 0], [0, 1e-6*np.random.random()]])
        s = -1.0/2 * x_mu * np.linalg.inv(cov) * x_mu.T
        res = 1.0/((2*np.pi) * np.sqrt(np.linalg.det(cov))) * (np.power(np.e, s))
    else:
        # s = -1.0/2 * x_mu * np.linalg.inv(cov) * x_mu.T
        # while(s > 1e10):
        #     s = s/2.
        # cov = np.matrix([[np.random.random()*10, 0], [0, np.random.random()*10]])
        # print "res^-1 =", (np.power(np.e, s)), "+", (2*np.pi) *np.sqrt(np.linalg.det(cov))
        # res = (np.power(np.e, s))/((2*np.pi) * np.sqrt(-np.linalg.det(cov)))
        mu = [np.matrix([np.random.random()*40., np.random.random()*40.]),
          np.matrix([np.random.random()*40., np.random.random()*40.]),
          np.matrix([np.random.random()*40., np.random.random()*40.])]
        x_mu = x-mu[0]
        cov += np.matrix([[np.random.random()*1e-6, 0], [0, 1e-6*np.random.random()]])
        s = -1.0/2 * x_mu * np.linalg.inv(cov) * x_mu.T
        res = 1.0/((2*np.pi) * np.sqrt(np.linalg.det(cov))) * (np.power(np.e, s))
        print 's =', s

    return res
def genData(dataNum=30):
    raw = np.loadtxt('haberman.txt',delimiter=',')
    # print raw
    x = raw[:, 1]
    y = raw[:, 3]
    labelMat = raw[:, 4]
    # print x, y, labelMat
    return x, y, labelMat

def GMM(dataMat, gammaMat, clusterNum=3):

    newLabelMat = []
    dataNum = len(dataMat)
    K = 3 #cluster number
    alpha = np.ones(3)/3

    mu = [np.matrix([np.random.random()*40., np.random.random()*40.]),
          np.matrix([np.random.random()*40., np.random.random()*40.]),
          np.matrix([np.random.random()*40., np.random.random()*40.])]
    cov = [np.matrix([[np.random.random()*20., 0.], [0., np.random.random()*20.]]),
           np.matrix([[np.random.random()*20., 0.], [0., np.random.random()*20.]]),
           np.matrix([[np.random.random()*20., 0.], [0., np.random.random()*20.]])]
    iterCnt = 100

#------------------------ E step begin: --------------------------------
    dataMat = np.matrix(dataMat)
    # gammaMat = np.matrix(gammaMat)
    Q = 0.
    preQ = 1.
    epsilon = 1e-12

    while True:
        print 'cov:', cov
        if(np.abs(Q-preQ)<epsilon):
            break
        preQ = Q
#------------------------ E step begin: --------------------------------
        for n in range(dataNum):
            sum = 1e-300
            for k in range(K):
                if(phi(dataMat[n], mu[k], cov[k]) > 1e300):
                    print '----------------------'
                    break
                sum += float(alpha[k]*phi(dataMat[n], mu[k], cov[k]))
                # print 'single:',alpha[k]*phi(dataMat[n], mu[k], cov[k])

            for k in range(K):
                print 'sum =', sum
                gammaMat[n][k] = float(alpha[k]*phi(dataMat[n], mu[k], cov[k])/sum)
                # print dataMat[n]
                # print 'gamma:', gammaMat[n][k]

#------------------------ E step end;   --------------------------------

#------------------------ M step begin: --------------------------------

        for k in range(K):
            sum_gamma = 1e-300
            sum_gammay = np.matrix([0., 0.])
            sum_gammaymu = np.matrix([[0., 0.], [0., 0.]])
            for j in range(dataNum):
                sum_gamma += (gammaMat[j][k])
                sum_gammay += gammaMat[j][k]*np.matrix(dataMat[j])
                # print sum_gammay
                sum_gammaymu += gammaMat[j][k]*(np.matrix(dataMat[j])-mu[k]).T *(np.matrix(dataMat[j])-mu[k])
            #-------\mu_k--------
            mu[k] = sum_gammay/sum_gamma
            # print 'mu =', mu

            #-------\cov_k--------
            cov[k]= sum_gammaymu/sum_gamma
            # print 'cov =', cov[k]

            #-------\alpha_k--------
            alpha[k] = sum_gamma/dataNum
            # print 'alphaqwq', alphaqwq

        iterCnt -= 1
        if iterCnt <= 0:
            break
        #
        sumQ = 0.
        for n in range(dataNum):
            sumqwq = 0.
            for k in range(clusterNum):
                sumqwq += alpha[k]*phi(dataMat[n], mu[k], cov[k])
                # print 'alphak:', alpha[k]
                # print 'phi:', phi(dataMat[n], mu[k], cov[k])
                # print 'sumqwq:', alpha[k]*phi(dataMat[n], mu[k], cov[k])
            sumQ += np.log(sumqwq)
            # print 'log', float(phi(dataMat[n], mu[k], cov[k]))
            # print 'sumQ:', sumQ

        Q = sumQ

        print 'Q:', Q
#------------------------ M step end;   --------------------------------

    for j in range(dataNum):
        tmp = findMax(gammaMat[j])
        newLabelMat.append(tmp)

    return newLabelMat

def findMax(gammaMat):
    a = gammaMat[0]
    res = 0
    if(a < gammaMat[1]):
        a = gammaMat[1]
        res = 1
    if(a < gammaMat[2]):
        a = gammaMat[2]
        res = 2
    # print 'res:', res
    return res

if __name__=="__main__":
    dataNum = 50
    dataMat, labelMat = [], []
    gammaMat = []
    # dataMat, labelMat = genData(dataNum)
    plt.figure(figsize=(12,6), dpi=80)
    plt.subplot(121)
    x, y, labelMat = genData(dataNum)
    x = x*12.
    y = y*30.
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        # print xi, yi
        if(labelMat[i] == 2):
            # labelMat[i] = 2
            plt.plot(xi, yi, color='#ffdf00', linestyle='None', marker='o')
        if(labelMat[i] == 1):
            # labelMat[i] = 1
            plt.plot(xi, yi, 'ro')
        if(labelMat[i] == 0):
            # labelMat[i] = 0
            plt.plot(xi, yi, color='#408d40', linestyle='None', marker='o')
    plt.axis('equal')

    for i in range(len(x)):
        tempMat = []
        tempMat.append(x[i])
        tempMat.append(y[i])
        dataMat.append(tempMat)
        gammaMat.append([0.333,0.333,0.333])

    originalDataMat = dataMat
    plt.subplot(122)
    newLabelMat = GMM(dataMat, gammaMat)

    for i in range(len(originalDataMat)):
        xi = originalDataMat[i][0]
        yi = originalDataMat[i][1]
        if(newLabelMat[i] == 2):
            plt.plot(xi, yi, color='#ffdf00', linestyle='None', marker='o')
        if(newLabelMat[i] == 1):
            plt.plot(xi, yi, 'ro')
        if(newLabelMat[i] == 0):
            plt.plot(xi, yi, color='#408d40', linestyle='None', marker='o')
    plt.axis('equal')
    # print dataMat
    plt.show()




