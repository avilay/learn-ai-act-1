import sklearn.datasets as ds
import sklearn.cross_validation as cv
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import sklearn.learning_curve as lc
import numpy as np
from sklearn.svm import SVC


def plot_learning_curve(title, train_sizes, train_scores, val_scores):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    # plot the training scores
    # axis=1 takes the mean/std of each row
    # Note: this is contrary to np.insert() which inserts a new col for axis=1 and a new row for axis=0
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r')
    # plot the validation scores
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g')
    plt.legend(['Train Scores', 'Val Scores'], loc='upper left')


def plot_validation_curve(title, xvals, xlabel, train_scores, val_scores):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    # plot the training scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    plt.fill_between(xvals, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.semilogx(xvals, train_scores_mean, color='r')
    # plot the validation scores
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    plt.fill_between(xvals, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.semilogx(xvals, val_scores_mean, color='g')
    plt.legend(['Train Scores', 'Val Scores'], loc='upper left')


def learning_curves():
    digits = ds.load_digits()
    X, y = digits.data, digits.target
    cviter = cv.ShuffleSplit(X.shape[0], n_iter=100, test_size=0.2)
    # If instead of ShuffleSplit I were to use KFold with shuffle set to True,
    # I would get test size roughly as n / n_folds. In case of digits data where n = 1797, the test size for each fold
    # would be around 18. This is too little data to get smooth train and test scores.
    # With ShuffleSplit I can control the test size.
    # cviter = cv.KFold(X.shape[0], n_folds=100, shuffle=True)
    _train_sizes = np.linspace(0.1, 1.0, 5)

    # learning_curve will internally call fit and score for each cv fold
    # train_scores is a (5, 100) matrix - one row for each training size, and 100 scores for each of the 100 folds.
    clf = GaussianNB()
    train_sizes, train_scores, val_scores = lc.learning_curve(clf, X, y, cv=cviter, train_sizes=_train_sizes)
    plot_learning_curve('Learning Curves (Naive Bayes)', train_sizes, train_scores, val_scores)

    clf = SVC(gamma=0.001)
    train_sizes, train_scores, val_scores = lc.learning_curve(clf, X, y, cv=cviter, train_sizes=_train_sizes)
    plot_learning_curve('Learning Curves (SVM, RBF kernel, $\gamma=0.001$)', train_sizes, train_scores, val_scores)

    plt.show()


def validation_curves():
    digits = ds.load_digits()
    X, y = digits.data, digits.target
    clf = SVC()
    param_range = np.logspace(-6, -1, 5)
    train_scores, val_scores = lc.validation_curve(clf, X, y, param_name='gamma', param_range=param_range)
    plot_validation_curve('Validation Curve with SVM', param_range, '$\gamma$', train_scores, val_scores)
    plt.show()

if __name__ == '__main__':
    learning_curves()
    # validation_curves()