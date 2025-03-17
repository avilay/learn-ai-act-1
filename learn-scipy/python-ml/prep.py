import sklearn.preprocessing as prep
import numpy as np


# Scaling is what Ng calls normalization. Here each column is subtracted by its mean and divided by the std
def scaling():
    X = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]], dtype=float)
    # mean of 1, 2, 3 is 1 and std is 0.82
    # so the cols will become (1 - 2) / 0.82 = - 1.22, (2 - 2) / 0.82 = 0, (3 - 2) / 0.82 = 1.22
    X_scaled = prep.scale(X)
    print(X_scaled)

    # When using scale I have to remember the means and stds of all cols so I can apply them to new data
    # To make this more convenient, use Scaler object instead.
    scaler = prep.StandardScaler().fit(X)
    # Now scaler has the mean and std of X
    print('Col means: {}'.format(scaler.mean_))
    print('Col stds: {}'.format(scaler.std_))
    X_new = np.array([[4, 4, 4], [5, 5, 5]], dtype=float)
    # X_new_scaled will have (4 - 2) / 0.82 = 2.44, (5 - 1) / 0.82 = 3.66
    X_new_scaled = scaler.transform(X_new)
    print(X_new_scaled)


# Normalizing is scaling each vector to have unit norm. Here each row is treated individually.
# Each element in the row is divided by the L2 norm or L1 norm (magnitude) of the row vector.
def normlizing():
    X = np.array([[1., -1., 2.],
                  [2., 0., 0.],
                  [0., 1., -1.]])
    # L2 magnitude of [1., -1., 2.] is 1/sqrt(6) = 0.41, -1/sqrt(6) = - 0.41, 2/sqrt(6) = 0.82
    X_norm = prep.normalize(X, norm='l2')
    print(X_norm)

    # Just like StandardScaler, there is a Normalizer object that can be used with Pipelines.
    normalizer = prep.Normalizer().fit(X)  # fit does nothing
    vec = normalizer.transform(np.array([[3., 2., 4.]]))
    print(vec)


# Binarizing is converting each element in a vector to 1s and 0s
# depending on whether they are above or below a certain threshold which defaults to 0.
# When given a matrix, it will work on each row vector
def binarizing():
    X = np.array([[100., -90., 2.], [-23., 1., 0], [-3., -20., 0.5]])
    print(prep.binarize(X))

    binarizer = prep.Binarizer(threshold=10).fit(X)  # Does nothing
    print(binarizer.transform(X))


# If y vector is given as a "vector" of strings, this will convert it to a numerical vector
def labeling():
    y = np.array(['paris', 'paris', 'tokyo', 'london'])
    labler = prep.LabelEncoder().fit(y)
    print(labler.classes_)
    print(labler.transform(['tokyo', 'tokyo', 'paris']))
    print(labler.inverse_transform([2, 2, 2]))


if __name__ == '__main__':
    # scaling()
    # normlizing()
    # binarizing()
    labeling()