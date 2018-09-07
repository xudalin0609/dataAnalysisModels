# -*- coding:utf-8 -*-
import numpy as np
from sklearn import datasets


class Kmean:

    def __init__(self, k):
        self.k = k
        self.labels = None

    def __distance(self, p1, p2):
        tem = np.sum((p1-p2)**2)
        return np.sqrt(tem)

    def __randCentrals(self):
        n_features = self.X.shape[1]
        self.centrals = np.zeros((self.k, n_features))
        for i in range(n_features):
            p_max, p_min = np.max(self.X[:, i]), np.min(self.X[:, i])
            self.centrals[:, i] = p_min + np.random.rand(1, self.k) * (p_max-p_min)

    def __converged(self, centrals1, centrals2):
        set1 = set([tuple(arr) for arr in centrals1])
        set2 = set([tuple(arr) for arr in centrals2])
        return set1 == set2

    def predict(self, X):
        self.X = X
        self.n = X.shape[0]
        self.__randCentrals()
        labels = np.zeros(self.n, dtype=np.int)
        converged = False
        test = 0
        while not converged:
            old_centrals = np.copy(self.centrals)
            # 迭代每个点
            for sample in range(self.n):
                min_dist, min_index = np.inf, -1
                for centralPointIndex in range(self.k):  # 迭代到每个中心的距离，保留最小距离
                    dist = self.__distance(self.centrals[centralPointIndex], X[sample])
                    if dist < min_dist:
                        min_dist = dist
                        labels[sample] = centralPointIndex
            '''
            更新中心坐标
            更新规则：
            这个中心下所有坐标的平均值作为新的中心
            '''

            for m in range(self.k):
                self.centrals[m] = np.mean(X[labels == m], axis=0)
            converged = self.__converged(old_centrals, self.centrals)
        self.labels = labels
        return self.centrals, self.labels


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    model = Kmean(3)
    centrals, labels = model.predict(X)

