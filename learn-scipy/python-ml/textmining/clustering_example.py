from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


# LSA is technique to reduce the vocab by eliminating similar words. It creates a matrix of word counts per paragraph,
# with words as the rows, and paragraphs as cols. It then uses SVD to reduce the number of rows.
def perform_latent_semantic_analysis(X):
    svd = TruncatedSVD(1000)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    X = lsa.fit_transform(X)
    exp_var = svd.explained_variance_ratio_.sum()
    print('Explained variance of SVD step: {}'.format(int(exp_var * 100)))
    return X


def main():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    # twenty = fetch_20newsgroups(subset='all', categories=categories)
    twenty = fetch_20newsgroups(subset='all', categories=categories, remove=('header', 'footer'))
    labels = twenty.target
    true_k = np.unique(labels).shape[0]  # The real number of clusters
    # Using TfidfVectorizer is equivalent to using a CountVectorizer followed by a TfidfTransformer
    vect = TfidfVectorizer(max_df=0.5, max_features=100000, min_df=2, stop_words='english', use_idf=True)
    X = vect.fit_transform(twenty.data)
    print('n_samples: {}, n_features: {}'.format(X.shape[0], X.shape[1]))
    # X = perform_latent_semantic_analysis(X)
    # print('After LSA - n_samples: {}, n_features: {}'.format(X.shape[0], X.shape[1]))
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
    km.fit(X)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels, sample_size=1000))
    print("\nTop terms per cluster:")

    # km.cluster_centers_ is a matrix of 4 X 10000 dims. Each row has the coordinates of one cluster
    # Each col is a unique word that appears in the cluster. The cell is the frequency of each word in that cluster.
    # In order to get the most frequently occuring words in each cluster, we need to sort each cluster
    # in descending order, hence the [:, ::-1]. But we are not interested in the actual frequencies,
    # we are more interested in the index number of the col which maps to a specific word. Hence the argsort.
    # Note, if LSA has been performed, then the below code will not give meaningful results. This is because the cols
    # of the centroids no longer match words, they are have been reduced to a mathematical construct.
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vect.get_feature_names()
    for i in range(true_k):
        print('\nCluster {}'.format(i))
        for ind in order_centroids[i, :10]:
            print(' {} '.format(terms[ind])),


if __name__ == '__main__': main()