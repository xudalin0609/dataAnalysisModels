# -*- coding:utf-8 -*-
'''
创建一个用于线性回归的类
相关的数学见博客https://blog.csdn.net/mumu0609/article/details/82154567
'''
import numpy as np


class LinearRegression:

    def __init__(self, n_iterations=3000, learning_rate=0.00004, regularization=None, gradient=True):
        '''
        :param n_iterations: 迭代次数
        :param learning_rate: 学习速率
        :param regularization: 是否使用正则
        :param gradient: 是否使用梯度下降
        '''
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        if regularization == None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    def __initialize_weights(self, n_features):
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)
        # print(w)

    def fit(self, X, y):
        '''
        :param X: 训练数据的自变量（特征）
        :param y: 训练数据的因变量（标签）
        :return: self
        '''
        m_samples, n_features = X.shape
        self.__initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
        self.train_errors = []
        if self.gradient == True:
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                loss = np.mean(0.5 * (y - y_pred) ** 2) + self.regularization(self.w)
                self.train_errors.append(loss)
                w_grad = X.T.dot(y_pred - y) + self.regularization.grad(self.w)
                print(w_grad)
                self.w = self.w + self.learning_rate * w_grad
        else:
            X = np.matrix(X)
            y = np.matrix(y)
            _coef = X.T.dot(X)
            _coef = _coef.I
            _coef = _coef.dot(X.T)
            self.w = _coef.dot(y)

    @property
    def error(self):
        return self.train_errors

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


if __name__ == '__main__':
    train_x = np.array(np.random.randint(0, 100, 100)).reshape(10, 10)
    # print(train_x)
    train_y = np.array([i for i in range(10)])
    model = LinearRegression()
    model.fit(train_x, train_y)
    print(model.error)
