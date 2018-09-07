import pandas as pd
import math


class DecisionTree:

    def __init__(self):
        self.entropy = None

    def generateData(self):
        trainSet = [['N', 0, 1, 0, 0],
                   ['Y', 0, 1, 1, 0],
                   ['Y', 1, 1, 1, 0],
                   ['N', 0, 1, 1, 1],
                   ['Y', 1, 1, 0, 1],
                   ['N', 1, 0, 1, 0],
                   ['Y', 1, 1, 0, 1]]
        data = pd.DataFrame(trainSet, columns=['Play', 'outlook', 'temperature', 'humidity', 'windy'])
        return data

    def calcShannonEnt(self, axis):
        featNum = {}
        for feac in self.data[axis]:
            if feac not in featNum.keys():
                featNum[feac] = 1
            else:
                featNum[feac] += 1
        featEnt = 0.0
        for feac in featNum.keys():  # 获得每个特征的所有类别
            prob = 0.0
            for i in self.lableClass:
                classNum = self.splitData(axis, feac, i)
                if classNum == 0:
                    continue
                prob -= classNum/featNum[feac] * math.log(classNum/featNum[feac], 2)
            featEnt += featNum[feac] / self.entireEnt * prob
        return featEnt, axis

    def splitData(self, axis, feac, value):
        splitedData = self.data[axis] == feac
        splitedData = self.data[splitedData]
        splitedData = splitedData[splitedData.iloc[:, 0]==value][axis].shape[0]
        return splitedData

    def calcAllEnt(self):
        numSample = self.data.shape[0]
        classNum = {}
        for label in self.data.iloc[:, 0]:
            if label not in classNum.keys():
                classNum[label] = 1
            else:
                classNum[label] += 1
        entireEnt = 0.0
        for label in classNum.keys():
            prob = classNum[label]/numSample * math.log(classNum[label]/numSample, 2)
            entireEnt -= prob
        return entireEnt

    def chooseBestEnt(self):
        featEnt = None
        self.entireEnt = self.calcAllEnt()
        for feature in self.data.columns[1:]:
            currentEnt, currentLabel = self.calcShannonEnt(feature)
            if featEnt == None:
                featEnt = self.entireEnt - currentEnt
                label = currentLabel
            elif featEnt < self.entireEnt - currentEnt:
                featEnt = self.entireEnt - currentEnt
                label = currentLabel
        return featEnt, label

    def cutData(self, label=None):
        if label == None:
            pass
        else:
            self.data = self.data.drop(label, axis=1)

    def createTree(self):
        bestEnt, label = self.chooseBestEnt()
        for i in self.data[label].unique():
            if self.data.iloc[:, 1:].shape[1] == 1:
                print(self.data)
                return 'N'
            self.cutData(label)
            self.myTree[label] = self.createTree()

    def majorityCnt(self):
        bestEnt, label = self.chooseBestEnt()


    def fit(self, data):
        self.myTree = {'temperature': {}}
        self.data = data
        self.lableClass = set(data.iloc[:, 0])
        bestEnt, label = self.chooseBestEnt()
        self.createTree()
        print(self.myTree)


if __name__ == '__main__':
    model = DecisionTree()
    data = model.generateData()
    model.fit(data)