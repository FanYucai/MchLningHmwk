# coding=utf-8

import numpy as np

def sigmoid(s):
    return 1.0/(1+np.power(np.e, -s))

def calh(theta, data):
    h = 0

    for i in range(len(data)):
        h += theta[i][0]*data[i]
    return h

def genDataMat(dataMat, originThetaNum):
    newDataMat = []
    dataNum = len(dataMat)
    for n in range(dataNum):
        newRow = []
        for i in range(originThetaNum):
            for j in range(i+1):
                newRow.append(dataMat[n][i]*dataMat[n][j])
        newDataMat.append(newRow)
    return newDataMat

def gradientDescent(dataMatIn, labelMatIn, sum_dataNum, dataNum, xNum, order=2, alpha=0.05, iter_times=100):
    thetaNum = (xNum+1)*(xNum+2)/2
    originThetaNum = 5

    theta = np.ones((thetaNum, 1))

    dataMat = genDataMat(dataMatIn, originThetaNum)

    dataList = dataMat
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMatIn).transpose()

    predLabelMat = []

    for iter in range(iter_times):
        afterSig = sigmoid(dataMat*theta)
        theta = theta - alpha * dataMat.transpose()*(afterSig-labelMat)


    thetaArray = np.array(theta)

    for i in range(sum_dataNum):
        res = calh(thetaArray, dataList[i])
        res = sigmoid(res)

        if(res > 0.5):
            predLabelMat.append(1)
        else:
            predLabelMat.append(0)

    return predLabelMat, thetaArray


def gradientDescentP(dataMatIn, labelMatIn, sum_dataNum, dataNum, xNum, order=2, alpha=0.01, iter_times=100):
    thetaNum = (xNum+1)*(xNum+2)/2
    originThetaNum = 5
    lam = 0.2

    theta = np.ones((thetaNum, 1))

    dataMat = genDataMat(dataMatIn, originThetaNum)

    dataList = dataMat
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMatIn).transpose()

    predLabelMat = []

    for iter in range(iter_times):
        afterSig = sigmoid(dataMat*theta)
        theta = theta - alpha * (dataMat.transpose()*(afterSig-labelMat)+lam*np.abs(theta))
        # print 'theta:', theta

    thetaArray = np.array(theta)

    for i in range(sum_dataNum):
        res = calh(thetaArray, dataList[i])
        res = sigmoid(res)

        if(res > 0.5):
            predLabelMat.append(1)
        else:
            predLabelMat.append(0)

    return predLabelMat, thetaArray

def getData():
    dataMat = []
    labelMat = []
    fr = open('data2.txt')
    i = 0
    scale = []
    for line in fr.readlines():
        i = i+1
        lineArr = line.strip().split(',')

        # if i==1:
        #     print len(lineArr)
        #     for iter in range(len(lineArr)):
        #         scale.append(float(lineArr[iter]))

        # dataMat.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        dataMat.append([1.0, float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])

        if(lineArr[0] == 'L'):
        # if(lineArr[4] == 'Iris-setosa'):
        # if(lineArr[0] == '1'):
            labelMat.append(1)
        else:
            labelMat.append(0)
    return int(i*0.9), i, dataMat, labelMat


if __name__=="__main__":
    xNum = 4
    train_dataNum, sum_dataNum, dataMat, labelMat = getData()
    predLabelMat, thetaArray = gradientDescent(dataMat, labelMat, sum_dataNum, train_dataNum, xNum , 2)
    corCnt = 0

    print "-------------------------------------------------------------------------"
    print "GD without penalty:\n"

    for i in range(train_dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt = corCnt + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]

    print "\n--->>>>>>>"
    print "validation:"
    print "--->>>>>>>\n"

    corCnt_v = 0
    for i in range(train_dataNum, sum_dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt_v = corCnt_v + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]

    print "\ntraining correctness is:", corCnt*1.0/train_dataNum*100, '%'
    print "validation correctness is:", corCnt_v*1.0/(sum_dataNum-train_dataNum)*100, '%'
    # print 'theta:', thetaArray

    print "-------------------------------------------------------------------------"
    print "GD with penalty:\n"

    predLabelMat, thetaArray_P = gradientDescentP(dataMat, labelMat, sum_dataNum, train_dataNum, xNum , 2)

    corCnt = 0
    for i in range(train_dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt = corCnt + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]

    print "\n--->>>>>>>"
    print "validation:"
    print "--->>>>>>>\n"

    corCnt_v = 0
    for i in range(train_dataNum, sum_dataNum):
        if(labelMat[i] == predLabelMat[i]):
            corCnt_v = corCnt_v + 1
        print 'original:', labelMat[i], '-> predict as:', predLabelMat[i]

    print "\ntraining correctness is:", corCnt*1.0/train_dataNum*100, '%'
    print "validation correctness is:", corCnt_v*1.0/(sum_dataNum-train_dataNum)*100, '%'
    # print 'theta:', thetaArray

