import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

from svm_plotter import SvmPlotter

def load_data():
    X = np.loadtxt('ex6data2_X.txt')
    y = np.loadtxt('ex6data2_y.txt')
    return (X, y)


def main():
    fignum = 1

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Check how the dummy estimator does on this data
    dmy = DummyClassifier(strategy='stratified')
    dmy.fit(X_train, y_train)
    dummy_error = 1 - dmy.score(X_test, y_test)
    print('DummyClassifier error: {}'.format(dummy_error))
    # It is 0.55 so it classifies half of the test data incorrectly

    # Check for skewness of data
    dmy = DummyClassifier(strategy='most_frequent')
    dmy.fit(X_train, y_train)
    dummy_error = 1 - dmy.score(X_test, y_test)
    print('DummyClassifier error: {}'.format(dummy_error))
    # It is 0.45 so it is still classifying half of the data incorrectly.
    # Data is not skewed so I can use the accuracy scoring method.

    # Plot the data with some usual params
    plt.figure(fignum)
    fignum += 1
    clf = SVC(C=1, gamma=2)
    clf.fit(X, y)
    pltr = SvmPlotter(clf, X, y)
    plt.figure(1)
    pltr.plot_data()
    # The fit is terrible
    
    # Get a feel for how the error varies with params
    plt.figure(fignum)
    fignum += 1
    pltr.plot_error_curve(Cs=[1,10,100,1000], gammas=[2, 4, 8, 16, 32])
    plt.figure(fignum)
    fignum += 1
    pltr.plot_validation_curve(Cs=[1,10,100,1000], gammas=[2, 4, 8, 16, 32])
    # Based on the graphs it seems like C=100, gamma=32 are the best params with an error of 0.004. 

    # Check with grid search.    
    tunable_params = [{
        'C': [1,10,100,1000],
        'gamma': range(1,33)
    }]
    gs = GridSearchCV(SVC(C=1, gamma=1, kernel='rbf'), tunable_params, cv=5, scoring='accuracy')
    ## by default SVC.score() uses the accuracy scoring method so it was not reqd to set the
    ## scoring param.
    gs.fit(X_train, y_train)
    # Grid search gave C=100, gamma=30 as the best params

    # Print the error report
    print("Best parameters set found on training set:")
    print(gs.best_estimator_)
    
    h = gs.predict(X_test)
    print(classification_report(y_test, h))
    
    test_error = 1 - gs.score(X_test, y_test)
    print('Test Error: {}'.format(test_error))

    # Plot the data with the best params
    plt.figure(fignum)
    fignum += 1
    clf = SVC(C=100, gamma=30)
    clf.fit(X,y)
    SvmPlotter(clf, X, y).plot_data()


if __name__ == '__main__':
    main()
    plt.show()