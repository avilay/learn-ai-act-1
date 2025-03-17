"""
Conceptually, there are two types of paramters that we need to figure out for any ML algo -
(a) The classification params
(b) The training hyper params
Take Logistic Regression as an example, the main learning algo learns the weights vector theta. However, we have to specify the regularization parameter lambda_ and if using gradient descent, then the learning rate alpha to the learning algo. In this case, theta vector are the classification params; lambda and alpha are the hyper params.
Using grid search along with cross validation, it is possible to learn a robust set of training hyper params.
Note: The classification params are automatically learnt by the ML algo anyways.
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

digits = datasets.load_digits()
X = digits.data
y = digits.target
m = X.shape[0]  # Number of samples
n = X.shape[1]  # Number of features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
params_grid = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
]

for score in ['precision', 'recall']:
    print('# Tuning hyper-parameters for {}'.format(score))
    print()

    clf = GridSearchCV(SVC(C=1), params_grid, cv=5, scoring='{}_weighted'.format(score))
    clf.fit(X_train, y_train)

    print('Best parameters set found on develoment set: ')
    print()
    print(clf.best_params_)
    print()
    print('Grid scores on development set:')
    print()
    for params, mean_score, scores in clf.grid_scores_:
        # scores is an array of score for each k-fold train/test cycle
        print('{:0.3f} (+/-{:0.03f}) for {}'.format(mean_score, scores.std() * 2, params))
    print()

    print('Detailed classification report')
    print()
    print('The model is trained on the full development set.')
    print('The scores are computed on the full evaluation set.')
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    # Prints the precision, recall, F-score for each class
    print(classification_report(y_true, y_pred))
    print()
