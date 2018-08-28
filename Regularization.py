# -*- coding:utf-8 -*-
'''
L1 L2 regularization
'''
import numpy as np


class l1Regularization:

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        loss = np.sum(np.fabs(w))
        return self.alpha * loss

    def grad(self, w):
        return self.alpha * np.sign(w)


class l2Regularization:

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        loss = W.dot(w)
        return self.alpha * 0.5 * float(loss)

    def grad(self, w):
        return self.alpha * w