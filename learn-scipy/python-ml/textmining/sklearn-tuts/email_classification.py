from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVC

# The data scturcture returned by fetch_20newsgroups. It assumes the following directory structure -
#     container_folder/
#             category_1_folder/
#                 file_1.txt
#                 file_2.txt
#                 ...
#                 file_42.txt
#             category_2_folder/
#                 file_43.txt
#                 file_44.txt
#                 ...
# Where container is either test or train, and category_1_folder would be alt.ahteism and so forth.as
# twenty_train = {
# 	target_names = ['category_1_folder', 'category_2_folder']
# 	target = [0, 0, 0, ..., 1, 1, ...]target = [0, 0, 0, ..., 1, 1, ...]
# 	filenames = [
# 		'../file_1.txt',
# 		'../file_2.txt',
# 		...,
# 		'../file_43.txt,
# 		'../file_44.txt
# 	]
# 	data = [
# 		'contents of file_1',
# 		'contents of file_2',
# 		...,
# 		'contents of file_43',
# 		'contents of file 44'
# 	]
# }
#


def main():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories)
    # eta0 is the learning rate - alpha in Eng's lectures
    # alpha is the regularization param (I think) - lambda in Eng's lectures
    clf = SGDClassifier(loss='hinge', penalty='l2', eta0=0.0001, alpha=0.0001, n_iter=5)
    # clf = SVC(kernel='linear')
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', clf)
    ])
    _ = text_clf.fit(twenty_train.data, twenty_train.target)

    twenty_test = fetch_20newsgroups(subset='test', categories=categories)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    accuracy = np.mean(predicted == twenty_test.target)
    print('Classifier accuracy: {}'.format(accuracy))
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
    conf_mat = metrics.confusion_matrix(twenty_test.target, predicted)
    print(conf_mat)

    # params = {
    #     'vect__ngram_range': [(1, 1), (1, 2)],
    #     'tfidf__use_idf': (True, False),
    #     # 'clf__C': [1, 2, 4, 8, 10],
    #     # 'clf__gamma': [0, 2, 4, 8]
    #     'clf__alpha': (0.0001, 0.001, 0.01, 0.1),
    #     'clf__eta0': (0.0001, 0.001, 0.01, 0.1)
    # }
    # gs_clf = GridSearchCV(text_clf, params, n_jobs=-1)
    # gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
    # best_params, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    # for param_name in sorted(params.keys()):
    #     print('{}: {}'.format(param_name, best_params[param_name]))

if __name__ == '__main__': main()