import numpy as np


def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = np.mat(np.zeros((numSamples, numSamples)))
    for i in range(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    return kernelMatrix


def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = np.mat(np.zeros((numSamples, 1)))

    if kernelType == 'linear':
        kernelValue = matrix_x * sample_x.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in xrange(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue


class SVM:

    def __init__(self, C, maxIter=2000, kernelOption = 'rbf', toler=0):
        self.C = C
        self.maxIter = maxIter
        self.kernerOption = kernelOption
        self.b = 0
        self.toler = toler

    def calcError(self, alpha_i):
        output_i = float(np.multiply(self.alphas, self.y).T * self.kernelMat[:, alpha_i] + self.b)
        error_i = output_i - float(self.y[alpha_i])
        return error_i

    def selectAlpha_j(self, alpha_i, error_i):
        self.errorCache[alpha_i] = [1, error_i]
        candidateAlphaList = np.nonzero(self.errorCache[:, 0].A)[0]
        maxStep = 0
        alpha_j = 0
        error_j = 0
        if len(candidateAlphaList) > 1:
            for k in candidateAlphaList:
                if k == alpha_i:
                    continue
                error_k = self.calcError(k)
                if np.abs(error_k - error_i) > maxStep:
                    maxStep = np.abs(error_k - error_i)
                    alpha_j = k
                    error_j = error_k
        else:
            alpha_j = alpha_i
            while alpha_j == alpha_i:
                alpha_j = int(np.random.uniform(0, self.m_samples))
            error_j = self.calcError(alpha_j)
        return alpha_j, error_j

    def fit(self, X, y, toler):
        self.X = X
        self.y = y
        self.m_samples, self.n_features = X.shape
        self.alphas = np.mat(np.zeros(self.m_samples, 1))
        self.kernelMat = calcKernelMatrix(self.X, self.kernerOption)
        self.errorCache = np.mat(np.zeros(self.m_samples, 2))
        entireSet = True
        alphaPairsChanged = 0
        iterCount = 0
        while (iterCount < self.maxIter) and ((alphaPairsChanged > 0) or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.m_samples):
                    alphaPairsChanged += self.innerLoop(i)
                print('--- iter:%d entire set, alpha pairs changed: %d' % (iterCount, alphaPairsChanged))
                iterCount += 1
            else:
                nonCoundAlphasList = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonCoundAlphasList:
                    alphaPairsChanged += self.innerLoop(i)
                    print('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
                    iterCount += 1

            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True

        return self

    def updateError(self, alpha_j):
        error = self.calcError(alpha_j)
        self.errorCache[alpha_j] = [1, error]

    def innerLoop(self, alpha_i):
        error_i = self.calcError(alpha_i)
        if (self.y[alpha_i] * error_i < -self.toler) and (self.alphas[alpha_i] < self.C) or \
                (self.y[alpha_i] * error_i > self.toler) and (self.alphas[alpha_i] > 0):
            alpha_j, error_j = self.selectAlpha_j(alpha_i, error_i)
            alpha_i_old = self.alphas[alpha_i].copy()
            alpha_j_old = self.alphas[alpha_j].copy()

            if self.y[alpha_i] != self.y[alpha_j]:
                L = np.max(0, self.alphas[alpha_j] - self.alphas[alpha_i])
                H = min(self.C, self.C + self.alphas[alpha_j] + self.alphas[alpha_i])
            else:
                L = max(0, self.alphas[alpha_j] + self.alphas[alpha_i] - self.C)
                H = min(self.C, self.alphas[alpha_j] + self.alphas[alpha_i])
            if L == H:
                return 0

            eta = 2.0 * self.kernelMat[alpha_i, alpha_j] - self.kernelMat[alpha_i, alpha_j] \
            - self.kernelMat[alpha_i, alpha_j]
            if eta >= 0:
                return 0
            self.alphas[alpha_j] -= self.y[alpha_j] * (error_i - error_j) / eta

            if self.alphas[alpha_j] > H:
                self.alphas[alpha_j] = H
            if self.alphas[alpha_j] < L:
                self.alphas[alpha_j] = L

            if np.abs(alpha_j_old - self.alphas[alpha_j]) < 0.00001:
                self.updateError(alpha_j)
                return 0

            self.alphas[alpha_i] += self.y[alpha_i] * self.y[alpha_j] \
                * (alpha_j_old - self.alphas[alpha_j])

            b1 = self.b - error_i - self.y[alpha_i] * (self.alphas[alpha_i] * alpha_j_old) \
                * self.kernelMat[alpha_i, alpha_i] - self.y[alpha_j] * (self.alphas[alpha_j] - alpha_j_old) \
                * self.kernelMat[alpha_i, alpha_j]
            b2 = self.b - error_j - self.y[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) \
                * self.kernelMat[alpha_i, alpha_j] - self.y[alpha_j] * (self.alphas[alpha_j]) \
                * self.kernelMat[alpha_j, alpha_j]

            if (0 < self.alphas[alpha_j]) and (self.alphas[alpha_i] < self.C):
                self.b = b1
            elif (0 < self.alphas[alpha_j]) and (self.alphas[alpha_j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            self.updateError(alpha_j)
            self.updateError(alpha_i)

            return 1
        else:
            return 0
