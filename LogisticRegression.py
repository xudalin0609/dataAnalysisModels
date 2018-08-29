# -*- coding:utf-8 -*-
import numpy as np
'''
创建一个用于实现逻辑回归的类
相关的数学推倒见博客
'''

class LogisticRegression:

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
        self.train_errors = []
        if regularization == None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    def sigmoid(self, X):
        return 1 / (1 + np.exp(X.dot(self.w)))

    def __initialize_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)
        # print(self.w.shape)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.__initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (1, m_samples))
        for i in range(self.n_iterations):
            y_pred = self.sigmoid(X).T
            loss = np.sum(y-y_pred)
            self.train_errors.append(loss)
            w_grad = (y-y_pred).dot(X)
            w_grad = w_grad.T
            self.w -= self.learning_rate*w_grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

    @property
    def error(self):
        return self.train_errors


