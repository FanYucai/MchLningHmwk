# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(s):
    return 1.0/(1+np.power(np.e, -s))

def genData(dataNum=30):
    xmean = np.random.random()*50
    print xmean
    ymean = np.random.random()*50
    print ymean
    xspan = np.random.random()*30
    yspan = np.random.random()*30
    print xspan, yspan
    mean = [xmean, ymean]
    cov = [[xspan, np.random.random()*40-20], [np.random.random()*40-20, yspan]]
    x, y = np.random.multivariate_normal(mean, cov, dataNum).T
    return x, y

def GMM(dataMat, labelMat):
    newLabelMat = []


    return newLabelMat

if __name__=="__main__":
    dataNum = 50
    dataMat, labelMat = [], []
    # dataMat, labelMat = genData(dataNum)
    plt.figure(figsize=(12,6), dpi=80)
    plt.subplot(121)
    x, y = genData(dataNum)
    plt.plot(x, y, 'co')
    for i in range(len(x)):
        tempMat = []
        tempMat.append(x[i])
        tempMat.append(y[i])
        dataMat.append(tempMat)
        labelMat.append(0)

    x, y = genData(dataNum)
    plt.plot(x, y, 'ro')
    for i in range(len(x)):
        tempMat = []
        tempMat.append(x[i])
        tempMat.append(y[i])
        dataMat.append(tempMat)
        labelMat.append(1)

    x, y = genData(dataNum)
    # plt.plot(x, y, color='#406606', linestyle='None', marker='.')
    plt.plot(x, y, 'k+')
    for i in range(len(x)):
        tempMat = []
        tempMat.append(x[i])
        tempMat.append(y[i])
        dataMat.append(tempMat)
        labelMat.append(2)
    plt.axis('equal')

    plt.subplot(122)
    # print dataMat
    # plt.axis([0, 100, 0, 100])
    newLabelMat = GMM(dataMat, labelMat)

    plt.show()




