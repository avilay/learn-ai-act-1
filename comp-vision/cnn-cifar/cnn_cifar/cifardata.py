import os.path as path
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

from .consts import *

label_encoder = LabelBinarizer().fit(list(range(10)))


def _unpickle(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    num_rows = len(batch['labels'])
    X = batch['data'].reshape((num_rows, 3, 32, 32)).transpose(0, 2, 3, 1)
    # X = batch['data']
    y = np.array(batch['labels'])
    filenames = batch['filenames']
    return X, y, filenames
    # return X, y


def all_valid():
    val_file = 'data_val'
    val_filepath = path.join(CIFAR_100_ROOT, val_file)
    return _unpickle(val_filepath)


def load_big_train_batch(batch_num):
    batch_file = 'data_batch_{}'.format(batch_num)
    batch_filepath = path.join(CIFAR_10_ROOT, batch_file)
    return _unpickle(batch_filepath)


def load_train_batch(batch_num):
    batch_file = 'data_batch_{}'.format(batch_num)
    batch_filepath = path.join(CIFAR_100_ROOT, batch_file)
    return _unpickle(batch_filepath)


def train_batches():
    for batch_num in range(1, 46):
        batch_file = 'data_batch_{}'.format(batch_num)
        batch_filepath = path.join(CIFAR_100_ROOT, batch_file)
        yield _unpickle(batch_filepath)


def test_batches():
    for batch_num in range(1, 11):
        test_batch_file = 'test_batch_{}'.format(batch_num)
        test_batch_filepath = path.join(CIFAR_100_ROOT, test_batch_file)
        yield _unpickle(test_batch_filepath)


def preprocess(X, y, fn):
    X_norm = X / 255
    Y = label_encoder.transform(y)
    return X_norm, Y, fn


def spot_check(X, filenames, num_samples, preprocessed=True, samples=None):
    m = X.shape[0]
    if not samples:
        samples = np.random.choice(np.arange(m), num_samples)
    print(samples)
    [print('{} - {}'.format(sample, filenames[sample])) for sample in samples]
    X_samples = []
    for i in samples:
        x = X[i]
        if preprocessed:
            x *= 255
        X_samples.append(x.astype(np.uint8))

    for i in range(num_samples):
        x = X_samples[i]
        fig = plt.subplot(1, num_samples, i+1)
        fig.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            labelleft='off',
            labelbottom='off'
        )
        fig.imshow(x)
    plt.show()
