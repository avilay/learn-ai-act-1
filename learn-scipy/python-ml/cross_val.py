import numpy as np
import sklearn.cross_validation as cv
import sklearn.datasets as ds
from sklearn import svm
from sklearn import metrics


def gen_data():
    X = np.array([0 + np.random.sample(3),
                  1 + np.random.sample(3),
                  2 + np.random.sample(3),
                  3 + np.random.sample(3),
                  4 + np.random.sample(3),
                  5 + np.random.sample(3),
                  6 + np.random.sample(3),
                  7 + np.random.sample(3),
                  8 + np.random.sample(3),
                  9 + np.random.sample(3)])
    y = np.random.randint(0, 3, X.shape[0])
    return X, y


def simple_split():
    """
    Simplest way of splitting data into train and test sets. No cv sets are possible with this.
    """
    X, y = gen_data()
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.4)
    print('X.shape = {}, y.shape = {}'.format(X.shape, y.shape))
    print('X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
    print('X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))


def cv_split():
    """
    Sample data rows can be split using various strategies.
    KFold serially (without shuffling) splits the data into k folds.
    StratifiedKFold serially splits the data into k folds but each fold has a represenatitive number of classes.
    Both of these can be used with the shuffle parameter.
    A ShuffleSplit iterator can be used for greater control like specifying the test/train split.
    """
    X, y = gen_data()

    kf = cv.KFold(len(y), n_folds=3)  # The test/train split will be approximately number of elements / number of folds
    print('Number of KFolds: {}'.format(len(kf)))
    for train_indexes, test_indexes in kf:
        print()
        print('Train indexes: {}, Test indexes: {}'.format(train_indexes, test_indexes))
        # X_train, y_train = X[train_indexes], y[train_indexes]
        # X_test, y_test = X[test_indexes], y[test_indexes]
        # print('X_train = {}'.format(X_train))
        # print('y_train = {}'.format(y_train))
        # print('X_test = {}'.format(X_test))
        # print('y_test = {}'.format(y_test))

    print()
    skf = cv.StratifiedKFold(y, n_folds=3)
    print('Number of StratifiedKFolds: {}'.format(len(skf)))
    for train_indexes, test_indexes in skf:
        print()
        print('Train indexes: {}, Test indexes: {}'.format(train_indexes, test_indexes))
        # X_train, y_train = X[train_indexes], y[train_indexes]
        # X_test, y_test = X[test_indexes], y[test_indexes]
        # print('X_train = {}'.format(X_train))
        # print('y_train = {}'.format(y_train))
        # print('X_test = {}'.format(X_test))
        # print('y_test = {}'.format(y_test))

    print()
    shuffkf = cv.KFold(len(y), n_folds=3, shuffle=True)
    print('Number of shuffled KFolds: {}'.format(len(shuffkf)))
    for train_indexes, test_indexes in shuffkf:
        print()
        print('Train indexes: {}, Test indexes: {}'.format(train_indexes, test_indexes))
        # X_train, y_train = X[train_indexes], y[train_indexes]
        # X_test, y_test = X[test_indexes], y[test_indexes]
        # print('X_train = {}'.format(X_train))
        # print('y_train = {}'.format(y_train))
        # print('X_test = {}'.format(X_test))
        # print('y_test = {}'.format(y_test))


def cv_scoring():
    iris = ds.load_iris()
    X = iris.data
    y = iris.target
    clf = svm.SVC(kernel='linear', C=1)

    # For each cv fold -
    #   this will fit (train) the classifier on the train set by calling estimator.fit(X_train, y_train) [Ref cross_validation.py#L1459]
    #   and get the score on the test set by calling scorer(estimator, X_test, y_test) [Ref cross_validation.py#L1534]
    #   returning a score for each fold. In the case below it will return 5 scores.
    scores = cv.cross_val_score(clf, X, y, cv=5)
    print('Got {} scores: {}'.format(len(scores), scores))
    scores = cv.cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
    print('Got {} scores: {}'.format(len(scores), scores))

    # Predict all the rows when they are not part of the training set
    y_pred = cv.cross_val_predict(clf, X, y, cv=10)
    accuracy = metrics.accuracy_score(y, y_pred)
    print('Accuracy is {}'.format(accuracy))


if __name__ == '__main__':
    # simple_split()
    # cv_split()
    cv_scoring()

