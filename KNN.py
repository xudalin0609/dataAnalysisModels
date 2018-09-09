import numpy as np


class KNN:

    def createDataSet():
        # create a matrix: each row as a sample
        group = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
        labels = ['A', 'A', 'B', 'B'] # four samples and two classes
        return group, labels

    def KNNClassify(self, newInput, dataSet, labels, k):
        m_samples = dataSet.shape[0]
        diff = np.tile(newInput, (m_samples, 1)) - dataSet
        squaredDiff = diff ** 2
        squarredDist = np.sum(squaredDiff, axis=1)
        distance = squarredDist ** 0.5
        sortedDistIndices = np.argsort(distance)
        classCount = {}
        for i in range(k):
            voteLabel = labels[sortedDistIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
            print('classCount:', classCount)
        maxCount = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        return maxIndex
