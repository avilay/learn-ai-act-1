import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import random

"""
There are two parameters that are used with SVM.
C determines how much overfitting I want. Higher it is the greater the penalty for getting an instance wrong,
and so SVM will generate a curve that will tend to overfit the data.

gamma determines how tightly the similarity function works. Higher gamma means tight similarity function,
i.e., as data points get farther the similarity function drops quickly to zero.
Smaller gamma means a more relaxed similarity function, i.e., similarity will drop gently.
This is inverse of sigma in Eng's ML class lectures.

A good strategy seems to be to increase gamma while checking performance on CV dataset and then increasing C.
"""


class SvmAnalyzer:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_pos = None
        self.X_neg = None
        self.clf = None

    def set_data(self, X, y):
        self.X = X
        self.y = y
        self.X_pos = np.array([self.X[i, :] for (i,), val in np.ndenumerate(self.y) if val == 1])
        self.X_neg = np.array([self.X[i, :] for (i,), val in np.ndenumerate(self.y) if val == 0])

    def _plot_decision_boundary(self):
        # print("Plotting decision boundary")
        tp = raw_input("Enter to continue.")
        x1plot = np.linspace(min(self.X[:,0]), max(self.X[:,0]), 100)
        x2plot = np.linspace(min(self.X[:,1]), max(self.X[:,1]), 100)
        X1, X2 = np.meshgrid(x1plot, x2plot)
        Z = self.clf.decision_function(np.c_[X1.ravel(), X2.ravel()])
        Z = Z.reshape(X1.shape)
        plt.contour(X1, X2, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-0.5, 0, 0.5])

    def vary_C(self, gamma):
        plt.title('gamma = {}'.format(gamma))
        for i, C in enumerate([1, 10, 100, 1000]):
            self.clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            self.clf.fit(self.X, self.y)
            plt.subplot(221 + i)
            plt.title("C = {}".format(C))
            self.plot()

    def vary_gamma(self, C):
        plt.title('C = {}'.format(C))
        for i, gamma in enumerate([2, 4, 8, 16]):
            self.clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            self.clf.fit(self.X, self.y)
            plt.subplot(221 + i)
            plt.title("Gamma = {}".format(gamma))
            self.plot()

    def vary_kernel(self, C, gamma):
        plt.title('C = {}, gamma = {}'.format(C, gamma))
        for i, kernel in enumerate(['linear', 'poly', 'rbf']):
            self.clf = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            self.clf.fit(self.X, self.y)
            plt.subplot(221 + i)
            plt.title(kernel)
            self.plot()

    def calc_error(self, X_cv, y_cv):
        h_cv = self.clf.predict(X_cv)
        correct = h_cv != y_cv
        return float(np.sum(correct))/correct.shape[0]

    def get_clf(self):
        return self.clf

    def plot(self):
        plt.plot(self.X_pos[:,0], self.X_pos[:,1], 'k+', self.X_neg[:,0], self.X_neg[:,1], 'yo')
        self._plot_decision_boundary()

    def learn(self, C, gamma):
        plt.title("C = {} Gamma = {}".format(C, gamma))
        self.clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        self.clf.fit(self.X, self.y)


