import os.path as path
import pickle
import numpy as np
from sklearn.utils import shuffle

DATAROOT = '/data/traffic-signs'


class TrafficSignsLoader:
    def __init__(self):
        self._train_file = path.join(DATAROOT, 'train.p')
        self._val_file = path.join(DATAROOT, 'valid.p')
        self._test_file = path.join(DATAROOT, 'test.p')

        self._batch_size = None

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def validation_all(self):
        with open(self._val_file, mode='rb') as f:
            val = pickle.load(f)
        X_val, y_val = val['features'], val['labels']
        return X_val, y_val

    def _batches(self, X, y):
        num_rows = (X.shape[0] // self._batch_size) * self._batch_size
        X = X[:num_rows]
        y = y[:num_rows]
        for i in range(0, num_rows, self._batch_size):
            X_batch = X[i:i+self._batch_size]
            y_batch = y[i:i+self._batch_size]
            yield X_batch, y_batch

    def validation_batches(self):
        X_val, y_val = self.validation_all()
        return self._batches(X_val, y_val)

    def train_all(self):
        with open(self._train_file, mode='rb') as f:
            train = pickle.load(f)
        X_train, y_train = train['features'], train['labels']
        X_train, y_train = shuffle(X_train, y_train)
        return X_train, y_train

    def train_batches(self):
        X_train, y_train = self.train_all()
        return self._batches(X_train, y_train)

    def test_all(self):
        with open(self._test_file, mode='rb') as f:
            test = pickle.load(f)
        X_test, y_test = test['features'], test['labels']
        return X_test, y_test

    def test_batches(self):
        X_test, y_test = self.test_all()
        return self._batches(X_test, y_test)
