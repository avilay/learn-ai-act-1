import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_decision_boundary(clfr):
    theta1, theta2 = clfr.coef_[0]
    theta0 = clfr.intercept_
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = [-((theta1*x) + theta0)/theta2 for x in xp]
    plt.title('C = {}'.format(clfr.C))
    plt.plot(X_pos[:, 0], X_pos[:, 1], 'k+', X_neg[:, 0], X_neg[:, 1], 'yo', xp, yp)


X = np.loadtxt('ex6data1_X.txt')
y = np.loadtxt('ex6data1_y.txt')
X_pos = np.array([X[i, :] for (i,), val in np.ndenumerate(y) if val == 1])
X_neg = np.array([X[i, :] for (i,), val in np.ndenumerate(y) if val == 0])

plt.figure(1)
plt.plot(X_pos[:, 0], X_pos[:, 1], 'k+', X_neg[:, 0], X_neg[:, 1], 'yo')


plt.figure(2)

for i, C in enumerate([1.0, 50.0, 100.0]):
    clf = svm.SVC(C=C, kernel='linear')
    clf.fit(X, y)
    plt.subplot(221 + i)
    plot_decision_boundary(clf)

plt.show()
