# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(s):
    return 1.0/(1+np.power(np.e, -s))

def calh(theta, thetaNum, data):
    h = 0
    for i in range(len(data)):
        h += theta[i][0]*data[i]
    return h

def genDataMat(dataMat, originThetaNum, thetaNum):
    newDataMat = []
    dataNum = len(dataMat)
    for n in range(dataNum):
        newRow = []
        for i in range(originThetaNum):
            for j in range(i+1):
                newRow.append(dataMat[n][i]*dataMat[n][j])
        newDataMat.append(newRow)
    print len(newDataMat[0])
    return newDataMat

def gradientDescent(dataMatIn, labelMatIn, dataNum, xNum, order=2, alpha=0.01, iter_times=200):
    thetaNum = 15
    originThetaNum = 5

    theta = np.ones((thetaNum, 1))

    dataMat = genDataMat(dataMatIn, originThetaNum, thetaNum)

    dataList = dataMat
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMatIn).transpose()

    predLabelMat = []

    for iter in range(iter_times):
        afterSig = sigmoid(dataMat*theta)
        theta = theta - alpha * dataMat.transpose()*(afterSig-labelMat)
        print 'theta:', theta

    thetaArray = np.array(theta)

    for i in range(dataNum):
        res = calh(thetaArray, thetaNum, dataList[i])
        res = sigmoid(res)

        if(res > 0.5):
            predLabelMat.append(1)
        else:
            predLabelMat.append(0)

    return predLabelMat


def gradientDescentP(dataMatIn, labelMatIn, dataNum, xNum, order=2, alpha=0.01, iter_times=200):
    thetaNum = 15
    originThetaNum = 5
    lam = 0.1

    theta = np.ones((thetaNum, 1))

    dataMat = genDataMat(dataMatIn, originThetaNum, thetaNum)

    dataList = dataMat
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMatIn).transpose()

    predLabelMat = []

    for iter in range(iter_times):
        afterSig = sigmoid(dataMat*theta)
        theta = theta - alpha * (dataMat.transpose()*(afterSig-labelMat)+lam*theta)
        print 'theta:', theta

    thetaArray = np.array(theta)

    for i in range(dataNum):
        res = calh(thetaArray, thetaNum, dataList[i])
        res = sigmoid(res)

        if(res > 0.5):
            predLabelMat.append(1)
        else:
            predLabelMat.append(0)

    return predLabelMat

def getData():
    dataMat = []
    labelMat = []
    fr = open('data2.txt')
    i = 0
    for line in fr.readlines():
        i = i+1
        lineArr = line.strip().split(',')
        # dataMat.append([1.0, float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])
        dataMat.append([1.0, float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])
        # if(lineArr[4] == 'Iris-setosa'):
        if(lineArr[0] == 'L'):
            labelMat.append(1)
        else:
            labelMat.append(0)
    return i, dataMat, labelMat

def getData_v():
    dataMat = []
    labelMat = []
    fr = open('data3.txt')
    i = 0
    for line in fr.readlines():
        i = i+1
        lineArr = line.strip().split(',')
        
        dataMat.append([1.0, float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])
        if(lineArr[0] == 'L'):
            labelMat.append(1)
        else:
            labelMat.append(0)
    return i, dataMat, labelMat


if __name__=="__main__":
    xNum = 4

    dataNum, dataMat, labelMat = getData()
    print len(dataMat), len(labelMat)
    predLabelMat = gradientDescent(dataMat, labelMat, dataNum, xNum , 2)
    corCnt = 0
    for i in range(dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt = corCnt + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]
    print "correctness is:", corCnt*1.0/dataNum*100, '%'
    dataNum, dataMat, labelMat = getData_v()
    for i in range(dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt = corCnt + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]
    print "correctness is:", corCnt*1.0/dataNum*100, '%'


    dataNum, dataMat, labelMat = getData_v()
    print len(dataMat), len(labelMat)
    predLabelMat = gradientDescentP(dataMat, labelMat, dataNum, xNum , 2)
    corCnt = 0
    for i in range(dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt = corCnt + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]
    print "correctness is:", corCnt*1.0/dataNum*100, '%'
    dataNum, dataMat, labelMat = getData()
    for i in range(dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt = corCnt + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]
    print "correctness is:", corCnt*1.0/dataNum*100, '%'

