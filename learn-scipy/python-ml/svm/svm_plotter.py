import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn.learning_curve import validation_curve


class SvmPlotter():
    def __init__(self, clf, X, y):
        self.clf = clf
        self.X = X
        self.y = y
        self.X_pos = np.array([self.X[i, :] for (i,), val in np.ndenumerate(self.y) if val == 1])
        self.X_neg = np.array([self.X[i, :] for (i,), val in np.ndenumerate(self.y) if val == 0])

    def plot_data(self):
        self.clf.fit(self.X, self.y)
        plt.plot(self.X_pos[:, 0], self.X_pos[:, 1], 'k+', self.X_neg[:, 0], self.X_neg[:, 1], 'yo')
        self._plot_decision_boundary()

    def plot_error_curve(self, Cs, gammas):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self.X, self.y, test_size=0.3, random_state=0)
        plots = {}
        for gamma in gammas:
            plots[gamma] = {'xplot': [], 'yplot': []}
            for C in Cs:
                clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
                clf.fit(X_train, y_train)
                error = 1 - clf.score(X_test, y_test)
                plots[gamma]['xplot'].append(C)
                plots[gamma]['yplot'].append(error)

        plt.xlabel('C')
        plt.ylabel('error')
        for gamma in plots:
            xplot = plots[gamma]['xplot']
            yplot = plots[gamma]['yplot']
            plt.plot(xplot, yplot, label='gamma = {}'.format(gamma))
        plt.legend()

    def plot_validation_curve(self, Cs, gammas):
        if len(gammas) > 6: raise ValueError('len gammas cannot be more than 6.')
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self.X, self.y, test_size=0.3, random_state=0)
        for i, gamma in enumerate(gammas):
            clf = svm.SVC(kernel='rbf', gamma=gamma)
            train_scores, validn_scores = validation_curve(clf, X_train, y_train, param_name='C', param_range=Cs,
                                                           scoring='accuracy')
            train_scores_mean = np.mean(train_scores, axis=1)
            validn_scores_mean = np.mean(validn_scores, axis=1)
            plt.subplot(321 + i)
            plt.title('gamma = {}'.format(gamma))
            plt.xlabel('C')
            plt.ylabel('score')
            plt.semilogx(Cs, train_scores_mean, 'r', label='Train')
            plt.semilogx(Cs, validn_scores_mean, 'b', label='Validn')

    def _plot_decision_boundary(self):
        # print("Plotting decision boundary")
        x1plot = np.linspace(min(self.X[:, 0]), max(self.X[:, 0]), 100)
        x2plot = np.linspace(min(self.X[:, 1]), max(self.X[:, 1]), 100)
        X1, X2 = np.meshgrid(x1plot, x2plot)
        Z = self.clf.decision_function(np.c_[X1.ravel(), X2.ravel()])
        Z = Z.reshape(X1.shape)
        plt.contour(X1, X2, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-0.5, 0, 0.5])


