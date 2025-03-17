import sklearn.datasets as ds
import sklearn.svm as svm
import sklearn.cross_validation as cv
import sklearn.metrics as metrics
import numpy as np
from sklearn.dummy import DummyClassifier


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


def use_scoring():
    iris = ds.load_iris()
    X, y = iris.data, iris.target

    # Set the scorer with the "scoring" parameter to cross_validation or grid_search modules.
    clf = svm.SVC(probability=True)
    scores = cv.cross_val_score(clf, X, y, scoring='log_loss')
    print(scores)

    # Or directly use one of the pre-built scorer objects
    y_pred = clf.predict(X)
    accuracy = metrics.accuracy_score(y, y_pred)
    print('Accuracy score: {}'.format(accuracy))

    print()
    print('Available scorers are -')
    print(metrics.SCORERS)


def cost(y_truth, y_pred):
    return np.abs(y_truth - y_pred).max()


def custom_scorers():
    # Use the make_scorer method to build scorer objects
    iris = ds.load_iris()
    X, y = iris.data, iris.target

    # Some built-in scorers like fbeta are parameterized, so they need to be initialized before used
    ftwo_scorer = metrics.make_scorer(metrics.fbeta_score, beta=2)
    clf = svm.SVC()
    scores = cv.cross_val_score(clf, X, y, scoring=ftwo_scorer)
    print(scores)

    # Or I can create a customized scorer from a core function that takes in y_truth and y_pred.
    # make_scorer returns a scorer function that wraps the core function. The scorer function takes in 3 params -
    # - the estimator, X, and y_truth. The scorer gets y_pred by calling estimator.predict(X) and then calls the core
    # function to get the final score. [Ref scorer.py#L305 --> scorer.py#L245]
    # The core function can either be a cost function where greater_is_bad or a gain function where greater_is_good.
    # A scorer is always a gain function. So if a cost function is given to make_scorer, we must set greater_is_better to False.
    # This will return a scorer that will return the negative of the core function.
    scorer = metrics.make_scorer(cost, greater_is_better=False)
    X, y = gen_data()
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print('Raw cost value: {}'.format(cost(y, y_pred)))
    print('Scorer value: {}'.format(scorer(clf, X, y)))


def reports():
    y = np.array([2, 0, 2, 2, 0, 1])
    y_pred = np.array([0, 0, 2, 2, 0, 2])

    # Confusion matrix is laid out in "matrix" form instead of "cartesian" form
    # i.e., rows start from the top, columns from the left.
    # element i, j is the number of samples that are i, but classified as j
    print('Confusion Matrix -')
    print(metrics.confusion_matrix(y, y_pred))

    # Classification report prints out the precision, recall, f1-score, etc.
    # The labels array is actually a map from the target values as the index number to the string label.
    # So labels[i] is the string name of class i
    labels = ['class-0', 'class-1', 'class-2']
    print(metrics.classification_report(y, y_pred, target_names=labels))


if __name__ == '__main__':
    # use_scoring()
    custom_scorers()
    # reports()