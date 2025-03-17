import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from random import randint


class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, m=8.0, t=0.0):
        self.m = m
        self.t = t
        self.categories = None

    def fit(self, X, y):
        self.categories = np.unique(y)

    def predict(self, X):
        c = len(self.categories) - 1
        u = X.shape[0]
        h = []
        for i in range(0, u):
            if (X[i, 0] * self.m + X[i, 1] * self.m) > self.t:
                h.append(1)
            else:
                h.append(0)
        return h


def load_data1():
    X = np.c_[(.4, -.7),
              (-1.5, -1),
              (-1.4, -.9),
              (-1.3, -1.2),
              (-1.1, -.2),
              (-1.2, -.4),
              (-.5, 1.2),
              (-1.5, 2.1),
              (1, 1),
              (1.3, .8),
              (1.2, .5),
              (.2, -2),
              (.5, -2.4),
              (.2, -2.3),
              (0, -2.7),
              (1.3, 2.1)].T

    y = []
    for i in range(0, X.shape[0]):
        if (X[i, 0] * 8 + X[i, 1] * 8) > 0:
            y.append(1)
        else:
            y.append(0)

    y = np.array(y)
    return (X, y)


X, y = load_data1()
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_dev)
print(y_dev)
clf = MyClassifier(m=2, t=2)
clf.fit(X_dev, y_dev)
h = clf.predict(X_test)
print()
print(X_test)
print(y_test)
print(h)
print(clf.score(X_test, y_test))

tunable_params = [{
    'm': np.arange(0, 5, 0.2),
    't': np.arange(0, 2, 0.2)
}]
gs = GridSearchCV(MyClassifier(), tunable_params, cv=3, scoring='accuracy')
gs.fit(X_dev, y_dev)
print('Best parameters set found on dev set:')
print(gs.best_estimator_)


